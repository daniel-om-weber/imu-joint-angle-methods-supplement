"""
Generate VQF orientation estimates and run OpenSim IMU IK for all subjects.

Follows the OpenSense calibration workflow:
1. Generate VQF orientations (full + mocap-aligned)
2. Run IMUPlacer calibration
3. Apply heading correction
4. Run IK with low feet weights

Usage:
    python generate_vqf_opensim.py
"""
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
from qmt import oriEstOfflineVQF

from utils import load_imu_data, get_sensor_mappings, write_orientations_sto, get_aligned_time_range
from constants import SUBJECTS

ROOT = Path(__file__).resolve().parent
SENSOR_NAMES = ['pelvis_imu', 'femur_r_imu', 'femur_l_imu',
                'tibia_r_imu', 'tibia_l_imu', 'calcn_r_imu']
SAMPLE_RATE = 100.0
SENSOR_TO_OPENSIM_ROTATION = -np.pi / 2  # -90° around X axis
IK_ACCURACY = 1e-6


def get_subject_path(subject_id):
    """Return the walking data path for a subject."""
    return ROOT / f'data/{subject_id}/walking'


def generate_vqf_orientations(subject_id, align_to_mocap=True,
                              output_name='walking_orientations.sto'):
    """Generate VQF orientations for all sensors and write to .sto file."""
    subject_path = get_subject_path(subject_id)
    mappings = get_sensor_mappings(subject_path / 'IMU' / 'myIMUMappings_walking.xml')
    imu_dir = subject_path / 'IMU' / 'xsens' / 'LowerExtremity'

    trim_start = 0
    trim_end = None
    if align_to_mocap:
        time_range = get_aligned_time_range(subject_path, SAMPLE_RATE)
        trim_start = time_range['imu_start']
        trim_end = time_range['imu_end']
        if trim_start > 0 or trim_end is not None:
            duration = (trim_end - trim_start) / SAMPLE_RATE if trim_end else 'unknown'
            print(f"  Aligning to GT: samples [{trim_start}:{trim_end}] ({duration:.1f}s)")

    quaternions = {}
    time = None

    for sensor_name in SENSOR_NAMES:
        sensor_id = mappings.get(sensor_name)
        if not sensor_id:
            print(f"  {sensor_name}: not mapped, skipping")
            continue

        sensor_id_clean = sensor_id.lstrip('_')
        imu_files = list(imu_dir.glob(f"*{sensor_id_clean}.txt"))
        if not imu_files:
            print(f"  {sensor_name}: file not found for {sensor_id}, skipping")
            continue

        imu_df = load_imu_data(imu_files[0])
        acc = imu_df[['Acc_X', 'Acc_Y', 'Acc_Z']].values[trim_start:trim_end]
        gyr = imu_df[['Gyr_X', 'Gyr_Y', 'Gyr_Z']].values[trim_start:trim_end]
        mag = imu_df[['Mag_X', 'Mag_Y', 'Mag_Z']].values[trim_start:trim_end]

        q = oriEstOfflineVQF(gyr, acc, mag=mag, params={'Ts': 1.0/SAMPLE_RATE})
        quaternions[sensor_name] = q
        print(f"  {sensor_name}: {len(q)} samples")

        if time is None:
            time = np.arange(len(q)) / SAMPLE_RATE

    if not quaternions:
        raise ValueError(f"No valid sensors found for {subject_id}")

    min_len = min(len(q) for q in quaternions.values())
    time = time[:min_len]
    quaternions = {name: q[:min_len] for name, q in quaternions.items()}

    output_path = subject_path / 'IMU' / 'vqf' / output_name
    write_orientations_sto(output_path, time, quaternions, SENSOR_NAMES, int(SAMPLE_RATE))
    print(f"  Output: {output_path} ({min_len} samples)")

    return output_path


def run_imu_placer(subject_id, orientations_sto, posed_model_path):
    """Run IMUPlacer to create calibrated model with VQF orientations."""
    import opensim as osim

    subject_path = get_subject_path(subject_id)
    output_dir = subject_path / 'IMU' / 'vqf'
    output_dir.mkdir(parents=True, exist_ok=True)
    calibrated_model_path = output_dir / 'model_Rajagopal2015_calibrated.osim'

    imu_placer = osim.IMUPlacer()
    imu_placer.set_model_file(str(posed_model_path))
    imu_placer.set_orientation_file_for_calibration(str(orientations_sto))
    imu_placer.set_base_imu_label('pelvis_imu')
    imu_placer.set_base_heading_axis('z')
    imu_placer.set_sensor_to_opensim_rotations(osim.Vec3(SENSOR_TO_OPENSIM_ROTATION, 0, 0))

    imu_placer.run(False)

    calibrated_model = imu_placer.getCalibratedModel()
    calibrated_model.printToXML(str(calibrated_model_path))

    print(f"  IMUPlacer output: {calibrated_model_path}")
    return calibrated_model_path


def apply_heading_correction(subject_id, orientations_sto, posed_model_path):
    """Apply heading correction to orientation data."""
    import opensim as osim

    subject_path = get_subject_path(subject_id)
    marker_ik_path = subject_path / 'Mocap' / 'ikResults' / 'walking_IK.mot'
    output_sto = subject_path / 'IMU' / 'vqf' / 'walking_orientations_hc.sto'

    model = osim.Model(str(posed_model_path))
    state = model.initSystem()
    model.realizePosition(state)

    marker_motion = osim.TimeSeriesTable(str(marker_ik_path))
    col_idx = marker_motion.getColumnIndex('pelvis_rotation')
    pelvis_rotation = marker_motion.getRowAtIndex(0)[col_idx]

    osense = osim.OpenSenseUtilities()
    oTable = osim.TimeSeriesTableQuaternion(str(orientations_sto))

    R_sensor = osim.Rotation()
    R_sensor.setRotationFromAngleAboutX(SENSOR_TO_OPENSIM_ROTATION)
    osense.rotateOrientationTable(oTable, R_sensor)

    heading_axis = osim.CoordinateDirection(osim.CoordinateAxis(2), 1)  # +Z
    correction_vec = osim.OpenSenseUtilities.computeHeadingCorrection(
        model, state, oTable, 'pelvis_imu', heading_axis)
    computed_correction = correction_vec.get(1) * 180 / np.pi

    angular_correction = computed_correction - pelvis_rotation

    oTable_final = osim.TimeSeriesTableQuaternion(str(orientations_sto))
    R_heading = osim.Rotation()
    R_heading.setRotationFromAngleAboutZ(np.radians(angular_correction))
    osense.rotateOrientationTable(oTable_final, R_heading)

    osim.STOFileAdapterQuaternion.write(oTable_final, str(output_sto))

    print(f"  Heading correction: {angular_correction:.2f}° (computed={computed_correction:.2f}°, pelvis={pelvis_rotation:.2f}°)")
    print(f"  Output: {output_sto}")
    return output_sto


def run_opensim_ik(subject_id, orientations_sto, model_path):
    """Run OpenSim IMU IK with low feet weights."""
    import opensim as osim

    subject_path = get_subject_path(subject_id)
    output_dir = subject_path / 'IMU' / 'vqf' / 'IKResults' / 'IKWithErrorsExtremeLowFeetWeights'
    output_dir.mkdir(parents=True, exist_ok=True)

    ik_tool = osim.IMUInverseKinematicsTool()
    ik_tool.set_model_file(str(model_path))
    ik_tool.set_orientations_file(str(orientations_sto))
    ik_tool.set_results_directory(str(output_dir))
    ik_tool.set_sensor_to_opensim_rotations(osim.Vec3(SENSOR_TO_OPENSIM_ROTATION, 0, 0))
    ik_tool.set_accuracy(IK_ACCURACY)

    weights = {
        'pelvis_imu': 1.0, 'femur_r_imu': 1.0, 'femur_l_imu': 1.0,
        'tibia_r_imu': 0.5, 'tibia_l_imu': 0.5,
        'calcn_r_imu': 0.01, 'calcn_l_imu': 0.01,
    }
    weight_set = ik_tool.upd_orientation_weights()
    for sensor, weight in weights.items():
        w = osim.OrientationWeight(sensor, weight)
        weight_set.cloneAndAppend(w)

    ik_tool.run()

    ori_basename = Path(orientations_sto).stem
    (output_dir / f'ik_{ori_basename}.mot').rename(output_dir / 'walking_IK.mot')
    (output_dir / f'ik_{ori_basename}_orientationErrors.sto').rename(
        output_dir / 'walking_orientationErrors.sto'
    )

    return output_dir / 'walking_IK.mot'


def process_subject(subject_id):
    """Run the full VQF + IK pipeline for one subject."""
    try:
        subject_path = get_subject_path(subject_id)
        posed_model = subject_path / 'IMU' / 'madgwick' / 'model_Rajagopal2015_posed.osim'

        print(f"\n{subject_id}:")

        full_sto = generate_vqf_orientations(subject_id, align_to_mocap=False,
                                             output_name='walking_orientations_full.sto')
        sto_path = generate_vqf_orientations(subject_id, align_to_mocap=True,
                                             output_name='walking_orientations.sto')

        calibrated_model = run_imu_placer(subject_id, full_sto, posed_model)
        corrected_sto = apply_heading_correction(subject_id, sto_path, posed_model)
        mot_path = run_opensim_ik(subject_id, corrected_sto, model_path=calibrated_model)

        print(f"  IK output: {mot_path}")

    except Exception as e:
        print(f"\n{subject_id}: ERROR - {e}")


def main():
    print(f"Running VQF + IK for: {', '.join(SUBJECTS)}")

    n_workers = min(len(SUBJECTS), cpu_count())
    with Pool(n_workers) as pool:
        pool.map(process_subject, SUBJECTS)


if __name__ == "__main__":
    main()
