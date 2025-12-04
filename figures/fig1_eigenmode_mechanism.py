#!/usr/bin/env python3
"""
Figure 1: Eigenmode Expansion Mechanism
Two-panel figure: eigenvalue spectrum + participation ratio schematic
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.linewidth': 1.2,
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

fig = plt.figure(figsize=(12, 5))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

# ==== Panel A: Eigenvalue spectrum as LINE PLOT (cleaner than bars) ====
ax1 = fig.add_subplot(gs[0])

n_modes = 40
mode_idx = np.arange(1, n_modes + 1)

# Eigenvalue spectra (as smooth curves)
eigenvalues_baseline = 12 * np.exp(-mode_idx / 4.5)
eigenvalues_psychedelic = 12 * np.exp(-mode_idx / 9)  # Slower decay = more modes above threshold

threshold = 1.2

# Plot as filled areas
ax1.fill_between(mode_idx, 0, eigenvalues_baseline, alpha=0.4, color='#3498db',
                  label='Baseline', step='mid')
ax1.fill_between(mode_idx, 0, eigenvalues_psychedelic, alpha=0.4, color='#e74c3c',
                  label='Psychedelic', step='mid')

# Threshold line
ax1.axhline(y=threshold, color='black', linestyle='--', linewidth=2, label='Activation threshold')

# Count active modes
n_active_baseline = np.sum(eigenvalues_baseline > threshold)
n_active_psychedelic = np.sum(eigenvalues_psychedelic > threshold)

# Remove confusing vertical bars - the filled areas already show the difference clearly

# Annotations
ax1.annotate(f'Baseline:\n{n_active_baseline} active\nmodes',
             xy=(n_active_baseline, threshold), xytext=(n_active_baseline + 5, 4),
             fontsize=10, color='#2874a6', fontweight='bold', ha='left',
             arrowprops=dict(arrowstyle='->', color='#2874a6', lw=1.5))

ax1.annotate(f'Psychedelic:\n{n_active_psychedelic} active\nmodes',
             xy=(n_active_psychedelic, threshold), xytext=(n_active_psychedelic + 3, 2.5),
             fontsize=10, color='#c0392b', fontweight='bold', ha='left',
             arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))

ax1.set_xlabel('Eigenmode Index (ranked by variance)', fontsize=11)
ax1.set_ylabel('Eigenvalue (variance explained)', fontsize=11)
ax1.set_title('A. Eigenmode Spectrum Expansion', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlim(0, 40)
ax1.set_ylim(0, 14)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# ==== Panel B: Participation Ratio Schematic ====
ax2 = fig.add_subplot(gs[1])

# Create two pie-chart style visualizations showing variance distribution
# Left: Baseline (concentrated), Right: Psychedelic (distributed)

ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_aspect('equal')
ax2.axis('off')

# Title
ax2.text(5, 9.5, 'B. Participation Ratio ($D_{eff}$)', fontsize=12, fontweight='bold',
         ha='center', va='top')

# Baseline pie (left side)
baseline_sizes = [50, 25, 12, 8, 5]  # 5 dominant modes
baseline_colors = ['#3498db', '#5dade2', '#85c1e9', '#aed6f1', '#d6eaf8']

ax2_left = fig.add_axes([0.52, 0.18, 0.18, 0.45])  # [left, bottom, width, height] - lowered pies
wedges1, _ = ax2_left.pie(baseline_sizes, colors=baseline_colors, startangle=90,
                           wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2))
ax2_left.text(0, -1.8, 'BASELINE', fontsize=10, fontweight='bold', ha='center', color='#2874a6')
ax2_left.text(0, -2.3, '$D_{eff}$ ≈ 3.6', fontsize=11, ha='center', color='#2874a6')
ax2_left.text(0, -2.8, '(concentrated)', fontsize=9, ha='center', color='#666')

# Psychedelic pie (right side) - more even distribution
psychedelic_sizes = [15, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]  # 11 active modes
psychedelic_colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(psychedelic_sizes)))

ax2_right = fig.add_axes([0.72, 0.18, 0.18, 0.45])  # lowered to match left pie
wedges2, _ = ax2_right.pie(psychedelic_sizes, colors=psychedelic_colors, startangle=90,
                            wedgeprops=dict(width=0.7, edgecolor='white', linewidth=1))
ax2_right.text(0, -1.8, 'PSYCHEDELIC', fontsize=10, fontweight='bold', ha='center', color='#c0392b')
ax2_right.text(0, -2.3, '$D_{eff}$ ≈ 9.2', fontsize=11, ha='center', color='#c0392b')
ax2_right.text(0, -2.8, '(distributed)', fontsize=9, ha='center', color='#666')

# Formula box - centered above the pie charts
formula_text = r'$D_{eff} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$'

ax2.text(5, 7.8, formula_text, fontsize=11, ha='center', va='center',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#ccc', linewidth=1, zorder=100))

plt.savefig('fig1_eigenmode_mechanism.pdf', format='pdf', bbox_inches='tight')
plt.savefig('fig1_eigenmode_mechanism.png', format='png', dpi=300, bbox_inches='tight')
print("Saved: fig1_eigenmode_mechanism.pdf and .png")
