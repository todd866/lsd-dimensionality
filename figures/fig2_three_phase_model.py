#!/usr/bin/env python3
"""
Figure 2: The Three-Phase Model of Psychedelic Dimensionality Dynamics
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.linewidth': 1.5,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def three_phase_curve(t):
    """Generate smooth three-phase curve using piecewise functions."""
    d_eff = np.zeros_like(t, dtype=float)

    # Parameters (in hours)
    t_onset = 0.5
    t_peak = 2.5
    t_offset = 8
    t_nadir = 36
    t_recover = 200

    # Peak and nadir values (relative to baseline=1)
    peak_val = 1.85
    nadir_val = 0.72

    for i, ti in enumerate(t):
        if ti < t_onset:
            # Pre-onset: baseline
            d_eff[i] = 1.0
        elif ti < t_peak:
            # Rise phase: smooth rise to peak
            progress = (ti - t_onset) / (t_peak - t_onset)
            d_eff[i] = 1.0 + (peak_val - 1.0) * (1 - np.cos(np.pi * progress)) / 2
        elif ti < t_offset:
            # Plateau/early decline
            progress = (ti - t_peak) / (t_offset - t_peak)
            d_eff[i] = peak_val - (peak_val - 1.2) * progress**0.5
        elif ti < t_nadir:
            # Refractory descent
            progress = (ti - t_offset) / (t_nadir - t_offset)
            d_eff[i] = 1.2 - (1.2 - nadir_val) * (1 - np.exp(-3 * progress))
        else:
            # Recovery
            progress = (ti - t_nadir) / (t_recover - t_nadir)
            progress = min(progress, 1.5)
            d_eff[i] = nadir_val + (1.0 - nadir_val) * (1 - np.exp(-2.5 * progress))

    return d_eff

# Create figure with more width
fig, ax = plt.subplots(figsize=(12, 5.5))

# Time array using log-ish spacing for better resolution at early times
t_early = np.linspace(0, 12, 200)
t_mid = np.linspace(12, 72, 100)
t_late = np.linspace(72, 336, 100)
t = np.concatenate([t_early, t_mid[1:], t_late[1:]])

d_eff = three_phase_curve(t)

# Shade phases FIRST (behind everything)
phase1_end = 8
phase2_end = 72

ax.axvspan(0, phase1_end, alpha=0.15, color='#27ae60', zorder=1)
ax.axvspan(phase1_end, phase2_end, alpha=0.15, color='#c0392b', zorder=1)
ax.axvspan(phase2_end, 336, alpha=0.15, color='#2980b9', zorder=1)

# Baseline
ax.axhline(y=1.0, color='#666666', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)

# Main curve
ax.plot(t, d_eff, 'k-', linewidth=3.5, zorder=5)

# Phase labels - positioned in CLEAR space within each zone
# Phase 1: small zone, put label at top LEFT corner
ax.text(1, 2.05, 'PHASE 1', ha='left', va='bottom', fontsize=12,
        fontweight='bold', color='#1e8449')
ax.text(1, 1.95, 'Overshoot', ha='left', va='top', fontsize=10,
        fontstyle='italic', color='#1e8449')

# Phase 2: larger zone, put label in center-bottom area
ax.text(35, 0.48, 'PHASE 2', ha='center', va='bottom', fontsize=12,
        fontweight='bold', color='#922b21')
ax.text(35, 0.48, 'Refractory', ha='center', va='top', fontsize=10,
        fontstyle='italic', color='#922b21')

# Phase 3: largest zone, put label in center
ax.text(180, 1.18, 'PHASE 3', ha='center', va='bottom', fontsize=12,
        fontweight='bold', color='#1a5276')
ax.text(180, 1.18, 'Recanalization', ha='center', va='top', fontsize=10,
        fontstyle='italic', color='#1a5276')

# Key annotations with arrows - positioned to NOT cross the curve
# 5-HT2A annotation - text to the right, arrow curves down to peak
ax.annotate('5-HT2A agonism\nDendritic gain ↑',
            xy=(2.5, 1.85), xytext=(25, 1.65),
            fontsize=9, ha='left', va='center',
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.5,
                          connectionstyle='arc3,rad=-0.3'))

# Receptor downregulation - text below curve, arrow points up to nadir
ax.annotate('Receptor\ndownregulation',
            xy=(36, 0.72), xytext=(55, 0.52),
            fontsize=9, ha='left', va='top',
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.5,
                          connectionstyle='arc3,rad=0.2'))

# Structural plasticity - text ABOVE curve, arrow points down
ax.annotate('Structural\nplasticity',
            xy=(120, 0.92), xytext=(100, 1.15),
            fontsize=9, ha='left', va='bottom',
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.5,
                          connectionstyle='arc3,rad=0.2'))

# "New landscape" annotation on far right
ax.annotate('Same $D_{eff}$,\nnew attractor\nlandscape',
            xy=(300, 1.0), xytext=(280, 1.30),
            fontsize=9, ha='center', va='bottom', color='#1a5276',
            arrowprops=dict(arrowstyle='->', color='#1a5276', lw=1.5))

# Format axes
ax.set_xlabel('Time post-administration', fontsize=12)
ax.set_ylabel('Effective Dimensionality ($D_{eff}$)', fontsize=12)
ax.set_xlim(0, 336)
ax.set_ylim(0.4, 2.2)

# Clean x-ticks with better spacing
ax.set_xticks([0, 8, 24, 72, 168, 336])
ax.set_xticklabels(['0', '8h', '24h', '3d', '1wk', '2wk'])

# Y-axis formatting
ax.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])

# Add baseline label
ax.text(330, 1.03, 'baseline', fontsize=9, color='#666', ha='right', va='bottom')

# Clean legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#27ae60', alpha=0.3, label='Phase 1: Acute expansion'),
    Patch(facecolor='#c0392b', alpha=0.3, label='Phase 2: Refractory compression'),
    Patch(facecolor='#2980b9', alpha=0.3, label='Phase 3: Recanalization'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)

ax.set_title('Three-Phase Model of Psychedelic Dimensionality Dynamics',
             fontsize=14, fontweight='bold', pad=15)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig2_three_phase_model.pdf', format='pdf', bbox_inches='tight')
plt.savefig('fig2_three_phase_model.png', format='png', dpi=300, bbox_inches='tight')
print("Saved: fig2_three_phase_model.pdf and .png")
