"""Generate method comparison barplot from RMSE results using mean."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

ROOT = Path(__file__).resolve().parent

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['mathtext.fontset'] = 'stix'

(ROOT / 'plots').mkdir(exist_ok=True)

# Methods to include and their display names
METHODS = {
    'MADGWICK': 'Madgwick + IK',
    'VQF-IK': 'VQF + IK',
    'vqf+olsson': 'VQF + Olsson',
    'weygers': 'Weygers',
}

knee_df = pd.read_csv(ROOT / 'results/knee_rmse_summary.csv', index_col='subject')
ankle_df = pd.read_csv(ROOT / 'results/ankle_rmse_summary.csv', index_col='subject')

# Exclude MEAN row to get per-subject values
knee_df = knee_df.drop('MEAN')
ankle_df = ankle_df.drop('MEAN')

knee_mean = [np.mean(knee_df[col]) for col in METHODS.keys()]
ankle_mean = [np.mean(ankle_df[col]) for col in METHODS.keys()]

# Compressed-scale parameters
THRESH = 15       # degrees: breakpoint between normal and compressed regions
COMPRESS = 0.15   # scale factor above threshold


def _compress_forward(y):
    y = np.asarray(y, dtype=float)
    return np.where(y <= THRESH, y, THRESH + (y - THRESH) * COMPRESS)


def _compress_inverse(y):
    y = np.asarray(y, dtype=float)
    return np.where(y <= THRESH, y, THRESH + (y - THRESH) / COMPRESS)


x = np.arange(len(METHODS))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_yscale('function', functions=(_compress_forward, _compress_inverse))

ax.bar(x - width/2, knee_mean, width, label='Knee', color='#3498db')
ax.bar(x + width/2, ankle_mean, width, label='Ankle', color='#e67e22')

ax.set_ylim(0, 60)

# Y-axis ticks: normal region + compressed region
normal_ticks = [0, 5, 10, 15]
compressed_ticks = [20, 25, 30, 35, 40, 45, 50,55]
all_ticks = normal_ticks + compressed_ticks
label_ticks = {0, 5, 10, 15, 30, 50}
ax.set_yticks(all_ticks)
ax.set_yticklabels([f'{v:.0f}' if v in label_ticks else '' for v in all_ticks])

# Manual gridlines with different styles per region
for v in normal_ticks:
    ax.axhline(y=v, color='grey', alpha=0.3, linestyle='--', linewidth=0.5)
for v in compressed_ticks:
    ax.axhline(y=v, color='grey', alpha=0.5, linestyle=':', linewidth=0.8)

# Threshold indicator
ax.axhline(y=THRESH, color='#888888', linewidth=1.2, linestyle='-', zorder=1)
ax.axhspan(THRESH, 60, color='#f0f0f0', alpha=0.3, zorder=0)

# Value labels on bars
for i, (knee_val, ankle_val) in enumerate(zip(knee_mean, ankle_mean)):
    for val, offset in [(knee_val, -width/2), (ankle_val, width/2)]:
        gap = 0.3 if val <= THRESH else 1.5
        ax.text(
            x[i] + offset, val + gap, f'{val:.1f}°',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# ax.set_title('Joint Angle Estimation Method Comparison',
            #  fontsize=16, fontweight='bold', pad=10)
ax.set_ylabel('Mean RMSE [°]', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(list(METHODS.values()), fontsize=11)
ax.tick_params(axis='x', length=0)
ax.legend(loc='upper right')

plt.savefig(ROOT / 'plots/method_comparison.pdf', bbox_inches='tight')
plt.savefig(ROOT / 'plots/method_comparison.svg', bbox_inches='tight')
plt.savefig(ROOT / 'plots/method_comparison.png', bbox_inches='tight')
print(f"Method comparison plot saved to {ROOT / 'plots/method_comparison.pdf'}")
plt.close()
