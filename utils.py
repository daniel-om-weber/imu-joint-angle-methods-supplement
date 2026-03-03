"""Utility functions for IMU joint angle estimation (data loading, signal alignment)."""
import json
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from scipy.signal import correlate, butter, filtfilt

from constants import FS

ROOT = Path(__file__).resolve().parent


# =============================================================================
# Data Loading
# =============================================================================

def load_mot(file_path):
    """Load OpenSim motion (.mot) file."""
    header_lines = 0
    with open(file_path, 'r') as f:
        for line in f:
            header_lines += 1
            if line.strip().startswith('time'):
                break
    df = pd.read_csv(file_path, sep='\t', skiprows=header_lines-1)
    df.columns = df.columns.str.strip()
    return df


def load_imu_data(file_path):
    """Load raw IMU data from .txt file (Xsens format)."""
    with open(file_path, 'r') as f:
        lines = f.readlines()

    header_line = None
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith('PacketCounter'):
            header_line = i
            data_start = i + 1
            break

    if header_line is None:
        raise ValueError(f"Could not find header in {file_path}")

    df = pd.read_csv(
        file_path,
        sep='\t',
        skiprows=data_start,
        names=lines[header_line].strip().split('\t')
    )
    return df


def _remove_nested_xml_comments(xml_str):
    """Remove XML comment blocks that contain nested comments."""
    import re
    while True:
        match = re.search(r'<!--(?:(?!-->).)*?<!--.*?-->', xml_str, re.DOTALL)
        if not match:
            break
        xml_str = xml_str[:match.start()] + xml_str[match.end():]
    return xml_str


def get_sensor_mappings(xml_path):
    """Parse XML to get sensor mappings {body_part: sensor_id}."""
    with open(xml_path, 'r') as f:
        content = _remove_nested_xml_comments(f.read())
    root = ET.fromstring(content)

    mappings = {}
    sensors = root.find('.//ExperimentalSensors')
    if sensors is not None:
        for sensor in sensors.findall('ExperimentalSensor'):
            name = sensor.get('name')
            model_name = sensor.find('name_in_model').text
            mappings[model_name] = name

    return mappings


def load_opensense_results(
    subject_path,  # path to subject data directory
    gt_column='knee_angle_r',  # column name to extract
    algorithm=None,  # 'xsens', 'madgwick', 'mahony' or None for all
    weighting='IKWithErrorsUniformWeights',  # IK weighting scheme
):
    """Load OpenSense IK results, returns dict mapping algorithm name to angles."""
    subject_path = Path(subject_path)
    algos = [algorithm] if algorithm else ['xsens', 'madgwick', 'mahony']
    results = {}

    for algo in algos:
        if algo not in ('xsens', 'madgwick', 'mahony'):
            print(f"Unknown algorithm: {algo}")
            continue

        path = subject_path / 'IMU' / algo / 'IKResults' / weighting / 'walking_IK.mot'

        if not path.exists():
            continue  # Silently skip missing algorithms

        df = load_mot(path)
        if gt_column not in df.columns:
            print(f"Column '{gt_column}' not found in {path}")
            continue

        results[algo] = df[gt_column].values

    return results


# =============================================================================
# Signal Alignment
# =============================================================================

def find_best_shift(
    est_signal,  # estimated signal array
    gt_signal,   # ground truth signal array
):
    """Find best alignment via cross-correlation, returns (offset, correlation)."""
    if np.std(est_signal) == 0 or np.std(gt_signal) == 0:
        return 0, 0.0
    if np.isnan(est_signal).any() or np.isnan(gt_signal).any():
        return 0, 0.0

    sig1 = (est_signal - np.mean(est_signal)) / (np.std(est_signal) + 1e-6)
    sig2 = (gt_signal - np.mean(gt_signal)) / (np.std(gt_signal) + 1e-6)

    corr = correlate(sig1, sig2, mode='full')
    lags = np.arange(-len(sig2) + 1, len(sig1))
    best_idx = np.argmax(corr)
    peak_corr = corr[best_idx] / min(len(sig1), len(sig2))
    # Negate to match align_signals convention (positive = trim gt)
    return -int(lags[best_idx]), float(peak_corr)


def _lowpass_filter(data, cutoff=5.0, fs=FS, order=4):
    """Low-pass filter for alignment signals (internal helper)."""
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def validate_offset(
    offset,      # alignment offset (positive = IMU starts before mocap)
    imu_len,     # length of IMU signal in samples
    mocap_len,   # length of mocap signal in samples
    min_overlap_fraction=0.5,  # minimum overlap as fraction of shorter signal
):
    """Check if offset produces valid overlap, returns (is_valid, overlap_samples, message)."""
    if offset > 0:  # IMU starts before mocap
        overlap = min(imu_len - offset, mocap_len)
    else:  # Mocap starts before IMU
        overlap = min(imu_len, mocap_len + offset)

    min_overlap = min_overlap_fraction * min(imu_len, mocap_len)

    if overlap < 0:
        return False, 0, "Offset produces no overlap"
    if overlap < min_overlap:
        return False, overlap, f"Overlap ({overlap}) below minimum ({min_overlap:.0f})"

    return True, overlap, f"Valid overlap: {overlap} samples ({overlap/FS:.1f} sec)"


def compute_raw_signal_offset(
    subject_path,  # path to subject data directory (e.g., 'data/Subject08/walking')
    fs=FS,         # sampling frequency
):
    """Compute IMU-mocap alignment offset via pelvis gyro correlation, returns (offset, correlation, error)."""
    subject_path = Path(subject_path)
    imu_dir = subject_path / 'IMU' / 'xsens' / 'LowerExtremity'
    mocap_path = subject_path / 'Mocap' / 'ikResults' / 'walking_IK.mot'
    mapping_path = subject_path / 'IMU' / 'myIMUMappings_walking.xml'

    # Load sensor mappings
    try:
        mappings = get_sensor_mappings(mapping_path)
    except Exception as e:
        return None, 0.0, f"Failed to load mappings: {e}"

    # Get pelvis sensor ID
    pelvis_sensor = mappings.get('pelvis_imu', '').lstrip('_')
    if not pelvis_sensor:
        return None, 0.0, "pelvis_imu not found in mappings"

    # Load pelvis IMU data
    try:
        pelvis_files = list(imu_dir.glob(f"*{pelvis_sensor}.txt"))
        if not pelvis_files:
            return None, 0.0, f"Pelvis IMU file not found for sensor {pelvis_sensor}"
        pelvis_df = load_imu_data(pelvis_files[0])
    except Exception as e:
        return None, 0.0, f"Failed to load pelvis IMU: {e}"

    # Load mocap data
    try:
        mocap_df = load_mot(mocap_path)
    except Exception as e:
        return None, 0.0, f"Failed to load mocap: {e}"

    pelvis_gyr = pelvis_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values

    # Correlate pelvis_Gyr_Z vs d/dt[pelvis_rotation] (correlation ~0.87)
    imu_signal = _lowpass_filter(pelvis_gyr[:, 2], cutoff=5.0, fs=fs)

    if 'pelvis_rotation' not in mocap_df.columns:
        return None, 0.0, "pelvis_rotation column not found in mocap"

    pelvis_rot = np.deg2rad(mocap_df['pelvis_rotation'].values)
    mocap_signal = _lowpass_filter(np.gradient(pelvis_rot) * fs, cutoff=5.0, fs=fs)

    # Compute offset with full correlation (searches all lags)
    offset, corr = find_best_shift(imu_signal, mocap_signal)

    # Validate offset (use negated offset for validation since find_best_shift already negates)
    is_valid, overlap, msg = validate_offset(-offset, len(imu_signal), len(mocap_signal))
    if not is_valid:
        return None, corr, f"Invalid offset: {msg}"

    return offset, corr, None


# =============================================================================
# Alignment Utilities
# =============================================================================

def get_aligned_time_range(
    subject_path,  # path to subject data directory (e.g., 'data/Subject08/walking')
    fs=FS,         # sampling frequency
):
    """Get IMU indices aligned with mocap, returns dict with imu_start, imu_end, gt_samples, offset."""
    subject_path = Path(subject_path)
    subject_id = subject_path.parent.name  # e.g., 'Subject08' from 'data/Subject08/walking'

    # Load ground truth to get duration
    mocap_path = subject_path / 'Mocap' / 'ikResults' / 'walking_IK.mot'
    gt_df = load_mot(mocap_path)
    gt_samples = len(gt_df)

    # Get alignment offset (cached or computed)
    offset = load_offset('raw_signal', subject_id, 'alignment')
    if offset is None:
        offset, corr, err = compute_raw_signal_offset(subject_path, fs)
        if err:
            print(f"Warning: Raw signal alignment failed ({err}), using zero offset")
            offset = 0
        else:
            save_offset('raw_signal', subject_id, 'alignment', offset)

    # Determine IMU range based on offset
    # Negative offset means IMU starts before mocap (common case)
    if offset < 0:
        imu_start = -offset  # Trim early IMU samples
        imu_end = imu_start + gt_samples
    else:
        # Mocap leads (rare): IMU starts at 0, but we'd trim GT instead
        # For VQF generation, we still start at 0 but process only gt_samples
        imu_start = 0
        imu_end = gt_samples

    return {
        'imu_start': imu_start,
        'imu_end': imu_end,
        'gt_samples': gt_samples,
        'offset': offset,
    }


def find_vqf_ik_file(subject_id, weighting='IKWithErrorsExtremeLowFeetWeights'):
    """Find generated VQF IK .mot file for subject. Returns Path or None."""
    base_path = ROOT / f'data/{subject_id}/walking/IMU/vqf/IKResults'
    # Check new structure first (with weighting subdirectory)
    candidate = base_path / weighting / 'walking_IK.mot'
    if candidate.exists():
        return candidate
    # Fall back to old structure (without weighting subdirectory)
    for name in ['walking_IK.mot', 'ik_walking_orientations.mot']:
        candidate = base_path / name
        if candidate.exists():
            return candidate
    return None


def load_offset(method, subject_id, gt_column):
    """Load cached offset from JSON. Returns int or None."""
    path = ROOT / 'offsets.json'
    if path.exists():
        offsets = json.loads(path.read_text())
        return offsets.get(f"{method}_{subject_id}_{gt_column}")
    return None


def save_offset(method, subject_id, gt_column, offset):
    """Save offset to JSON cache."""
    path = ROOT / 'offsets.json'
    offsets = json.loads(path.read_text()) if path.exists() else {}
    offsets[f"{method}_{subject_id}_{gt_column}"] = int(offset)  # Convert numpy int to Python int
    path.write_text(json.dumps(offsets, indent=2))


# =============================================================================
# OpenSim File Writing
# =============================================================================

def write_orientations_sto(
    output_path,   # path for output .sto file
    time,          # time array (N,)
    quaternions,   # dict mapping sensor_name -> quaternions (N, 4) in w,x,y,z
    sensor_names,  # list of sensor names in desired column order
    data_rate=100,  # sampling rate in Hz
):
    """Write quaternion orientations to OpenSim .sto format."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(f'DataRate={data_rate:.6f}\n')
        f.write('DataType=Quaternion\n')
        f.write('version=3\n')
        f.write('OpenSimVersion=4.4\n')
        f.write('endheader\n')

        # Header row - only include sensors that have data
        available_sensors = [s for s in sensor_names if s in quaternions]
        f.write('time\t' + '\t'.join(available_sensors) + '\n')

        # Data rows
        for i in range(len(time)):
            row = [f'{time[i]}']
            for name in available_sensors:
                q = quaternions[name][i]  # w, x, y, z
                row.append(f'{q[0]},{q[1]},{q[2]},{q[3]}')
            f.write('\t'.join(row) + '\n')
