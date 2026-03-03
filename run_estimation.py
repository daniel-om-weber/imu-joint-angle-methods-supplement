"""
Run 4 joint angle estimation methods (knee and ankle) and produce RMSE summary CSVs.

Usage:
    python run_estimation.py --joint knee --method all --subject Subject08
    python run_estimation.py --joint ankle --method madgwick_ik
    python run_estimation.py --joint knee --method all --subject all
    python run_estimation.py --joint all --method all --subject all
"""
import numpy as np
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd

from utils import (
    load_imu_data, get_sensor_mappings,
    load_opensense_results, find_vqf_ik_file,
    get_aligned_time_range, load_mot
)
from constants import FS, SUBJECTS
from methods import run_vqf_olsson_heading_corrected, run_weygers

ROOT = Path(__file__).resolve().parent


def _eval_method(name, est, gt, errors_dict):
    """Evaluate estimated angles against ground truth."""
    n = min(len(est), len(gt))
    est, gt = est[:n], gt[:n]
    error = np.abs(gt - est)
    errors_dict[name] = error
    print(f"{name} - RMSE: {np.sqrt(np.mean(error**2)):.2f} deg")


# Joint configuration
JOINTS = {
    'knee': {
        'proximal_sensor': 'femur_r_imu',
        'distal_sensor': 'tibia_r_imu',
        'gt_column': 'knee_angle_r'
    },
    'ankle': {
        'proximal_sensor': 'tibia_r_imu',
        'distal_sensor': 'calcn_r_imu',
        'gt_column': 'ankle_angle_r',
    },
}


def prepare_data(
    joint_name,              # 'knee' or 'ankle'
    subject_id='Subject08',  # subject identifier (e.g., 'Subject08')
):
    """Load and prepare IMU data, returns dict with acc/gyr arrays, fs, gt, paths, and alignment info."""
    joint_config = JOINTS[joint_name]
    subject_path = ROOT / f'data/{subject_id}/walking'
    imu_path = subject_path / 'IMU' / 'xsens' / 'LowerExtremity'
    fs = float(FS)

    # Get sensor mappings from XML
    mappings = get_sensor_mappings(subject_path / 'IMU' / 'myIMUMappings_walking.xml')
    prox_id = mappings.get(joint_config['proximal_sensor'])
    dist_id = mappings.get(joint_config['distal_sensor'])
    if not prox_id or not dist_id:
        raise ValueError(f"Could not find sensor IDs for {joint_name}")

    # Load IMU data
    prox_df = load_imu_data(list(imu_path.glob(f"*{prox_id}.txt"))[0])
    dist_df = load_imu_data(list(imu_path.glob(f"*{dist_id}.txt"))[0])

    acc_prox = prox_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyr_prox = prox_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values
    acc_dist = dist_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values
    gyr_dist = dist_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values

    # Load ground truth
    gt_df = load_mot(subject_path / 'Mocap' / 'ikResults' / 'walking_IK.mot')
    gt = gt_df[joint_config['gt_column']].values

    # Store original GT for precomputed methods (OpenSense, VQF-OpenSim)
    gt_original = gt

    # Get aligned time range using centralized utility
    time_range = get_aligned_time_range(subject_path, int(fs))
    offset = time_range['offset']
    imu_start = time_range['imu_start']
    imu_end = time_range['imu_end']
    print(f"Alignment offset: {offset} samples ({offset/fs:.2f} sec)")

    # Trim IMU data to aligned range
    acc_prox, gyr_prox = acc_prox[imu_start:imu_end], gyr_prox[imu_start:imu_end]
    acc_dist, gyr_dist = acc_dist[imu_start:imu_end], gyr_dist[imu_start:imu_end]

    # Truncate to common length (all arrays must match GT)
    n = min(len(acc_prox), len(acc_dist), len(gt))
    acc_prox, gyr_prox = acc_prox[:n], gyr_prox[:n]
    acc_dist, gyr_dist = acc_dist[:n], gyr_dist[:n]
    gt = gt[:n]

    print(f"Aligned data length: {n} samples ({n/fs:.1f} sec)")

    return {
        'acc_prox': acc_prox,
        'gyr_prox': gyr_prox,
        'acc_dist': acc_dist,
        'gyr_dist': gyr_dist,
        'fs': fs,
        'gt': gt,                   # Aligned GT for IMU methods
        'gt_original': gt_original, # Original GT for precomputed methods
        'alignment_offset': offset, # Raw signal alignment offset
        'subject_path': subject_path,
        'joint_config': joint_config,
        'subject_id': subject_id,
    }


def process_vqf_olsson(data, errors_dict):
    """Run VQF+Olsson+Heading Correction and add errors to dict."""
    print("\n=== VQF+Olsson ===")
    angle_deg = run_vqf_olsson_heading_corrected(
        data['acc_prox'], data['gyr_prox'], data['acc_dist'], data['gyr_dist'], data['fs']
    )
    _eval_method('vqf+olsson', angle_deg, data['gt'], errors_dict)


def _find_calibrated_model(subject_path):
    """Find a calibrated .osim model file for a subject, searching vqf first."""
    imu_dir = subject_path / 'IMU'
    candidates = (imu_dir / d / 'model_Rajagopal2015_calibrated.osim'
                  for d in ('vqf', 'madgwick', 'mahony', 'xsens'))
    return next((c for c in candidates if c.exists()), None)


def process_weygers(data, errors_dict):
    """Run Weygers KF with model axis mode."""
    print("\n=== Weygers ===")
    joint = 'knee' if 'knee' in data['joint_config']['gt_column'] else 'ankle'

    model_path = _find_calibrated_model(data['subject_path'])
    if model_path is None:
        print("No calibrated .osim model found, skipping")
        return

    angle_deg, _, _, jhat, q_rel = run_weygers(
        data['acc_prox'], data['gyr_prox'],
        data['acc_dist'], data['gyr_dist'],
        data['fs'],
        joint=joint,
        model_path=model_path,
        prox_imu=data['joint_config']['proximal_sensor'],
        dist_imu=data['joint_config']['distal_sensor'],
    )

    print(f"Joint axis: [{jhat[0]:.3f}, {jhat[1]:.3f}, {jhat[2]:.3f}]")
    _eval_method('weygers', angle_deg, data['gt'], errors_dict)


def process_madgwick_ik(data, errors_dict):
    """Load Madgwick+IK results and add errors to dict."""
    print("\n=== Madgwick+IK ===")
    results = load_opensense_results(
        data['subject_path'],
        data['joint_config']['gt_column'],
        algorithm='madgwick',
        weighting='IKWithErrorsExtremeLowFeetWeights'
    )
    if not results:
        print("No Madgwick+IK results found")
        return
    angle_deg = next(iter(results.values()))
    _eval_method('MADGWICK', angle_deg, data['gt_original'], errors_dict)


def process_vqf_ik(data, errors_dict):
    """Load VQF IK results using raw signal alignment (same as IMU methods)."""
    vqf_file = find_vqf_ik_file(data['subject_id'])
    if not vqf_file:
        print("\n=== VQF-IK: No file found ===")
        return

    print("\n=== VQF-IK ===")
    vqf_angle = load_mot(vqf_file)[data['joint_config']['gt_column']].values

    # Pre-aligned VQF-IK has ~60k samples matching mocap, unaligned has ~144k
    offset = data['alignment_offset']
    if len(vqf_angle) >= len(data['gt']) * 1.5 and offset < 0:
        vqf_angle = vqf_angle[-offset:]

    _eval_method('VQF-IK', vqf_angle, data['gt'], errors_dict)


METHOD_DISPATCH = {
    'vqf_olsson': process_vqf_olsson,
    'weygers': process_weygers,
    'madgwick_ik': process_madgwick_ik,
    'vqf_ik': process_vqf_ik,
}


def run_single_subject(joint, method, subject_id):
    """Run estimation on a single subject and return errors dict."""
    print(f"\n{'='*60}")
    print(f"Processing {subject_id} - {joint} joint")
    print(f"{'='*60}")

    try:
        data = prepare_data(joint, subject_id)
    except Exception as e:
        print(f"Error loading data for {subject_id}: {e}")
        return subject_id, {}

    errors_dict = {}
    methods = METHOD_DISPATCH if method == 'all' else {method: METHOD_DISPATCH[method]}
    for fn in methods.values():
        fn(data, errors_dict)

    return subject_id, errors_dict


def run_all_subjects(joint, method, workers=None):
    """Run estimation on all subjects in parallel."""
    print(f"\nRunning {method} on all subjects for {joint} joint...")
    print(f"Subjects: {', '.join(SUBJECTS)}")
    print(f"Workers: {workers or 'auto (CPU count)'}\n")

    results = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_single_subject, joint, method, subj): subj
            for subj in SUBJECTS
        }
        for future in as_completed(futures):
            subj = futures[future]
            try:
                _, errors = future.result()
                results[subj] = errors
            except Exception as e:
                print(f"Error processing {subj}: {e}")
                results[subj] = {}
    return results


def print_summary_table(results, joint):
    """Print RMSE summary table and save to CSV."""
    # Collect all methods from results
    methods = set()
    for errors in results.values():
        methods.update(errors.keys())
    methods = sorted(methods)

    if not methods:
        print("No results to summarize.")
        return

    # Build rows with RMSE values
    rows = []
    for subj in sorted(results.keys()):
        row = {'subject': subj}
        for m in methods:
            if m in results[subj] and len(results[subj][m]) > 0:
                row[m] = np.sqrt(np.mean(results[subj][m]**2))
            else:
                row[m] = np.nan
        rows.append(row)

    # Add mean row
    mean_row = {'subject': 'MEAN'}
    for m in methods:
        vals = [r[m] for r in rows if not np.isnan(r.get(m, np.nan))]
        mean_row[m] = np.mean(vals) if vals else np.nan  # type: ignore[assignment]
    rows.append(mean_row)

    df = pd.DataFrame(rows)

    # Print table
    print("\n" + "="*80)
    print(f"RMSE Summary - {joint.capitalize()} Joint (degrees)")
    print("="*80)
    print(df.to_string(index=False, float_format='%.2f'))

    # Save to CSV
    csv_path = ROOT / 'results' / f'{joint}_rmse_summary.csv'
    df.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"\nResults saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run 4 joint angle estimation methods and produce RMSE summary CSVs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--joint', type=str, default='knee', choices=['knee', 'ankle', 'all'],
                        help='Joint to estimate (default: knee)')
    parser.add_argument('--method', type=str, default='all',
                        choices=[*METHOD_DISPATCH, 'all'],
                        help='Estimation method (default: all)')
    parser.add_argument('--subject', type=str, default='Subject08',
                        help='Subject ID or "all" for all subjects (default: Subject08)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()

    (ROOT / 'results').mkdir(exist_ok=True)

    joints = ['knee', 'ankle'] if args.joint == 'all' else [args.joint]

    for joint in joints:
        if args.subject == 'all':
            results = run_all_subjects(joint, args.method, args.workers)
            print_summary_table(results, joint)
        else:
            _ = run_single_subject(joint, args.method, args.subject)


if __name__ == "__main__":
    main()
