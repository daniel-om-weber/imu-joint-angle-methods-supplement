# Multi-IMU Joint Angle Estimation Comparison

Compares four IMU-based joint angle estimation methods against motion capture ground truth for knee and ankle joints using dual-IMU sensor configurations. Produces `plots/method_comparison.pdf` — a grouped bar chart of mean RMSE across subjects.

## Methods

| Key | Display Name | Description |
|-----|-------------|-------------|
| `MADGWICK` | Madgwick + IK | Madgwick orientation filter through OpenSim inverse kinematics |
| `VQF-IK` | VQF + IK | VQF orientation filter through OpenSim inverse kinematics |
| `vqf+olsson+heading_correction` | VQF + Olsson | VQF orientation + Olsson hinge joint axis with heading correction |
| `weygers` | Weygers | Kalman filter with gravity frame constraints + model-based joint axis |

## Reproduction

```bash
# 1. Install dependencies
pip install .

# 2. Download dataset (~5.5 GB from SimTK)
python download_simtk_dataset.py
# If the script fails, download manually from https://simtk.org/plugins/datashare/index.php?group_id=2164
# and extract into the `data/` directory.

# 3. (Optional) Generate VQF + IK results — requires OpenSim
conda install -c opensim-org opensim   # not available via pip
python generate_vqf_opensim.py

# 4. Run all methods on all subjects
python run_estimation.py --joint all --method all --subject all

# 5. Generate figure
python create_comparison_figure.py
```

This produces `results/{knee,ankle}_rmse_summary.csv` and `plots/method_comparison.pdf`.

Step 3 is optional and only needed for the VQF + IK baseline. The other three methods (Madgwick + IK, VQF + Olsson, Weygers) work without OpenSim. Madgwick + IK results are precomputed in the dataset.

## Data

Download places data in `data/SubjectXX/walking/`:

```
data/Subject08/walking/
├── IMU/
│   ├── xsens/LowerExtremity/*.txt        # Raw IMU data (100 Hz)
│   ├── myIMUMappings_walking.xml          # Sensor-to-segment mappings
│   ├── madgwick/IKResults/                # Precomputed Madgwick IK results
│   └── vqf/IKResults/                    # Precomputed VQF IK results
└── Mocap/
    └── ikResults/walking_IK.mot           # Motion capture ground truth
```

Subjects: Subject02, Subject03, Subject04, Subject08.

## Dependencies

Core (pip-installable): `numpy`, `pandas`, `scipy`, `matplotlib`, `qmt`, `dfjimu`, `tqdm`, `requests`

Optional: `opensim` (conda only — for step 3 preprocessing)
