"""VQF + Olsson joint angle estimation method."""
import numpy as np
import qmt


def run_vqf_olsson(
    acc_prox,  # proximal accelerometer (N, 3)
    gyr_prox,  # proximal gyroscope (N, 3)
    acc_dist,  # distal accelerometer (N, 3)
    gyr_dist,  # distal gyroscope (N, 3)
    fs,        # sampling frequency in Hz
):
    """Estimate joint angle using VQF + Olsson, returns (angle_deg, jhat_prox, jhat_dist, q_rel, q_prox, q_dist)."""
    # Estimate orientations using VQF
    q_prox = qmt.oriEstOfflineVQF(gyr_prox, acc_prox, params={'Ts': 1.0/fs})
    q_dist = qmt.oriEstOfflineVQF(gyr_dist, acc_dist, params={'Ts': 1.0/fs})

    # Estimate joint axes using Olsson method
    jhat_prox, jhat_dist = qmt.jointAxisEstHingeOlsson(
        acc_prox, acc_dist, gyr_prox, gyr_dist
    )
    jhat_prox, jhat_dist = jhat_prox.flatten(), jhat_dist.flatten()

    # Calculate relative quaternion and joint angle
    q_rel = qmt.qmult(qmt.qinv(q_prox), q_dist)
    angle_deg = np.degrees(qmt.quatProject(q_rel, jhat_prox)['projAngle'])

    return angle_deg, jhat_prox, jhat_dist, q_rel, q_prox, q_dist


def run_vqf_olsson_heading_corrected(
    acc_prox,  # proximal accelerometer (N, 3)
    gyr_prox,  # proximal gyroscope (N, 3)
    acc_dist,  # distal accelerometer (N, 3)
    gyr_dist,  # distal gyroscope (N, 3)
    fs,        # sampling frequency in Hz
):
    """Estimate joint angle using VQF + Olsson with heading drift correction, returns angle_deg."""
    Ts = 1.0 / fs

    # Get base VQF+Olsson results (raw orientations and estimated joint axes)
    _, jhat_prox, jhat_dist, _, q_prox, q_dist = run_vqf_olsson(
        acc_prox, gyr_prox, acc_dist, gyr_dist, fs
    )

    # Apply heading correction to raw VQF orientations
    # headingCorrection returns: (quat2Corr, delta, deltaFilt, rating, state)
    # Only the distal quaternion is corrected; proximal remains unchanged
    t = qmt.timeVec(N=q_prox.shape[0], Ts=Ts)  # type: ignore[union-attr]
    q_dist_corr, *_ = qmt.headingCorrection(
        gyr1=gyr_prox, gyr2=gyr_dist,
        quat1=q_prox, quat2=q_dist,
        t=t,
        joint=jhat_prox,  # Use estimated joint axis (shape (3,) for 1D joint)
        jointInfo={},
        estSettings={'constraint': 'euler_1d'}
    )

    # Calculate relative quaternion from corrected orientations
    q_rel_corr = qmt.qmult(qmt.qinv(q_prox), q_dist_corr)

    # Use swing-twist decomposition for proper angle extraction around joint axis
    angle_deg = np.degrees(qmt.quatProject(q_rel_corr, jhat_prox)['projAngle'])

    return angle_deg
