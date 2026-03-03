"""Weygers et al. Kalman filter with gravity frame constraints for dual-IMU joint angle estimation."""
import xml.etree.ElementTree as ET

import numpy as np
import qmt

from dfjimu.mekf_acc import mekf_acc
from dfjimu import estimate_lever_arms


def _euler_xyz_to_rotmat(euler):
    """Convert OpenSim intrinsic XYZ Euler angles (radians) to rotation matrix."""
    rx, ry, rz = euler
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def parse_osim_calibration(
    model_path,   # path to calibrated .osim file
    joint_name,   # joint name substring to match (e.g. 'knee_r')
    prox_imu,     # proximal IMU frame name (e.g. 'femur_r_imu')
    dist_imu,     # distal IMU frame name (e.g. 'tibia_r_imu')
):
    """Parse calibrated .osim model for IMU-to-joint calibration frames, returns dict."""
    tree = ET.parse(model_path)
    root = tree.getroot()

    # Find the joint matching joint_name (CustomJoint or PinJoint)
    joint_el = None
    joint_type = None
    for tag in ('CustomJoint', 'PinJoint'):
        for jt in root.iter(tag):
            if joint_name in jt.get('name', ''):
                joint_el = jt
                joint_type = tag
                break
        if joint_el is not None:
            break
    if joint_el is None:
        raise ValueError(f"No joint matching '{joint_name}' in {model_path}")

    # Extract parent and child offset frame orientations from the joint
    parent_orient = child_orient = None
    for pof in joint_el.iter('PhysicalOffsetFrame'):
        name = pof.get('name', '')
        orient_el = pof.find('orientation')
        if orient_el is None:
            continue
        orient = np.array([float(v) for v in orient_el.text.split()])
        # Parent frame is listed first (socket_parent_frame), child second
        if parent_orient is None:
            parent_orient = orient
        else:
            child_orient = orient

    if parent_orient is None or child_orient is None:
        raise ValueError(f"Could not find parent/child offset frames in joint '{joint_name}'")

    # Extract rotation axis: CustomJoint has SpatialTransform, PinJoint rotates about Z
    if joint_type == 'PinJoint':
        rot_axis = np.array([0.0, 0.0, 1.0])
    else:
        rot_axis = np.array([1.0, 0.0, 0.0])  # default X-axis
        for ta in joint_el.iter('TransformAxis'):
            if ta.get('name') == 'rotation1':
                axis_el = ta.find('axis')
                if axis_el is not None:
                    rot_axis = np.array([float(v) for v in axis_el.text.split()])
                break

    # Find IMU PhysicalOffsetFrame orientations (top-level, not inside joint)
    imu_orients = {}
    for pof in root.iter('PhysicalOffsetFrame'):
        name = pof.get('name', '')
        if name in (prox_imu, dist_imu):
            orient_el = pof.find('orientation')
            if orient_el is not None:
                imu_orients[name] = np.array([float(v) for v in orient_el.text.split()])

    if prox_imu not in imu_orients:
        raise ValueError(f"IMU frame '{prox_imu}' not found in {model_path}")
    if dist_imu not in imu_orients:
        raise ValueError(f"IMU frame '{dist_imu}' not found in {model_path}")

    return {
        'R_prox_proxIMU': _euler_xyz_to_rotmat(imu_orients[prox_imu]),
        'R_dist_distIMU': _euler_xyz_to_rotmat(imu_orients[dist_imu]),
        'R_parent_offset': _euler_xyz_to_rotmat(parent_orient),
        'R_child_offset': _euler_xyz_to_rotmat(child_orient),
        'rot_axis': rot_axis / np.linalg.norm(rot_axis),
    }

# Noise parameters
_Q_COV = np.ones(6) * 1e-2              # process noise diagonal
_R_DIAG = 2.0 * 0.35**2 * 10           # measurement noise (2 * cov_lnk_scale)
_P_INIT_DIAG = 1.0                      # initial covariance diagonal


def _pre_align_imu(acc, gyr, R_body_to_imu):
    """Rotate IMU data from sensor frame to body frame."""
    return acc @ R_body_to_imu.T, gyr @ R_body_to_imu.T


def _model_joint_angle(q_rel, calib):
    """Calculate joint angle from body-frame q_rel using offset-frame chain (no R_s2b)."""
    A = calib['R_parent_offset'].T
    B = calib['R_child_offset']
    R_rel = qmt.quatToRotMat(q_rel)
    R_joint = np.einsum('ij,njk,kl->nil', A, R_rel, B)
    q_joint = qmt.quatFromRotMat(R_joint)
    return np.degrees(qmt.quatProject(q_joint, calib['rot_axis'])['projAngle'])


def run_weygers(
    acc_prox,          # proximal accelerometer (N, 3) or (3, N)
    gyr_prox,          # proximal gyroscope (N, 3) or (3, N)
    acc_dist,          # distal accelerometer (N, 3) or (3, N)
    gyr_dist,          # distal gyroscope (N, 3) or (3, N)
    fs,                # sampling frequency in Hz
    r1=None,           # lever arm 1, auto-estimated if None
    r2=None,           # lever arm 2, auto-estimated if None
    joint=None,        # 'knee' or 'ankle' for model mode
    model_path=None,   # path to calibrated .osim for model mode
    prox_imu=None,     # proximal IMU frame name for model mode
    dist_imu=None,     # distal IMU frame name for model mode
):
    """Estimate joint angle using Weygers KF with gravity constraints, returns (angle_deg, r1, r2, jhat, q_rel)."""
    # Ensure shape is (N, 3)
    if acc_prox.shape[1] != 3:
        acc_prox, gyr_prox = acc_prox.T, gyr_prox.T
        acc_dist, gyr_dist = acc_dist.T, gyr_dist.T

    # Pre-align IMU data to body frame using calibration
    if model_path is None or prox_imu is None or dist_imu is None:
        raise ValueError("run_weygers requires model_path, prox_imu, and dist_imu")

    joint_name = joint or 'knee_r'
    if '_r' not in joint_name and '_l' not in joint_name:
        joint_name += '_r'
    calib = parse_osim_calibration(model_path, joint_name, prox_imu, dist_imu)
    acc_prox, gyr_prox = _pre_align_imu(acc_prox, gyr_prox, calib['R_prox_proxIMU'])
    acc_dist, gyr_dist = _pre_align_imu(acc_dist, gyr_dist, calib['R_dist_distIMU'])

    # Estimate lever arms if not provided
    if r1 is None or r2 is None:
        r1, r2 = estimate_lever_arms(gyr_prox, gyr_dist, acc_prox, acc_dist, fs)

    # Run MEKF-acc (in body frame)
    q1_all, q2_all = mekf_acc(
        gyr_prox, gyr_dist, acc_prox, acc_dist, r1, r2, fs,
        np.array([1.0, 0, 0, 0]), _Q_COV, _R_DIAG, _P_INIT_DIAG,
    )

    # Compute relative quaternion (in body frame)
    q_rel = qmt.qmult(qmt.qinv(q1_all), q2_all)

    # Model-based angle calculation
    angle_deg = _model_joint_angle(q_rel, calib)
    jhat = calib['R_parent_offset'] @ calib['rot_axis']
    jhat = jhat / np.linalg.norm(jhat)
    return angle_deg, r1, r2, jhat, q_rel
