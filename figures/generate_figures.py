#!/usr/bin/env python3
"""
Generate publication-quality figures for psychedelics dimensionality paper.
Target: Translational Psychiatry
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from scipy import stats

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.linewidth': 1.0,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color scheme for compounds
COLORS = {
    'Psilocybin': '#7B68EE',  # Medium slate blue
    'LSD': '#FF6B6B',          # Coral red
    'Ketamine': '#4ECDC4',     # Teal
    'Tiagabine': '#95A5A6',    # Gray
    'Placebo': '#BDC3C7',      # Light gray
    'Drug': '#E74C3C',         # Red
}


def fig1_eigenmode_mechanism():
    """Figure 1: Eigenmode expansion mechanism."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel A: Eigenvalue spectra
    ax = axes[0]
    modes = np.arange(1, 21)
    # Baseline: steep decay
    baseline = 10 * np.exp(-0.3 * modes)
    # Psychedelic: flatter decay (more modes above threshold)
    psychedelic = 8 * np.exp(-0.15 * modes)

    ax.semilogy(modes, baseline, 'b-', lw=2, label='Baseline', marker='o', markersize=4)
    ax.semilogy(modes, psychedelic, 'r-', lw=2, label='Psychedelic', marker='s', markersize=4)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, label='Activation threshold')
    ax.set_xlabel('Eigenmode index')
    ax.set_ylabel('Variance (log scale)')
    ax.set_title('A. Eigenvalue Spectra')
    ax.legend(loc='upper right', frameon=False)
    ax.set_xlim(0, 21)
    ax.set_ylim(0.01, 15)

    # Panel B: Participation ratio illustration
    ax = axes[1]
    # Baseline: concentrated on few modes
    baseline_weights = np.array([0.6, 0.25, 0.1, 0.03, 0.02])
    baseline_weights = baseline_weights / baseline_weights.sum()
    # Psychedelic: distributed across modes
    psychedelic_weights = np.array([0.25, 0.22, 0.2, 0.18, 0.15])
    psychedelic_weights = psychedelic_weights / psychedelic_weights.sum()

    x = np.arange(1, 6)
    width = 0.35
    ax.bar(x - width/2, baseline_weights, width, label='Baseline', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, psychedelic_weights, width, label='Psychedelic', color='indianred', alpha=0.8)

    # Calculate D_eff for each
    d_eff_baseline = 1 / np.sum(baseline_weights**2)
    d_eff_psychedelic = 1 / np.sum(psychedelic_weights**2)

    ax.set_xlabel('Eigenmode')
    ax.set_ylabel('Variance fraction')
    ax.set_title('B. Variance Distribution')
    ax.legend(loc='upper right', frameon=False)
    ax.set_xticks(x)

    # Add D_eff annotations
    ax.text(0.05, 0.95, f'$D_{{eff}}$ = {d_eff_baseline:.1f}', transform=ax.transAxes,
            fontsize=9, color='steelblue', va='top')
    ax.text(0.05, 0.85, f'$D_{{eff}}$ = {d_eff_psychedelic:.1f}', transform=ax.transAxes,
            fontsize=9, color='indianred', va='top')

    # Panel C: Dendritic mechanism schematic
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('C. Dendritic Gain Mechanism')

    # Draw pyramidal neuron schematic
    # Soma
    soma = plt.Circle((5, 3), 0.8, color='lightgray', ec='black', lw=1.5)
    ax.add_patch(soma)

    # Apical dendrite
    ax.plot([5, 5], [3.8, 7.5], 'k-', lw=2)

    # Dendritic branches
    for dy, dx in [(6.5, -1.2), (6.5, 1.2), (7.5, -0.8), (7.5, 0.8)]:
        ax.plot([5, 5+dx], [dy, dy+0.5], 'k-', lw=1.5)

    # Basal dendrites
    for angle in [-30, -60, -120, -150]:
        rad = np.radians(angle)
        ax.plot([5, 5 + 1.5*np.cos(rad)], [3 - 0.8, 3 - 0.8 + 1.5*np.sin(rad)], 'k-', lw=1.5)

    # Axon
    ax.plot([5, 5], [3-0.8, 0.5], 'k-', lw=2)
    ax.annotate('', xy=(5, 0.3), xytext=(5, 0.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 5-HT2A receptors on apical dendrite
    for y in [5, 5.5, 6, 6.5, 7]:
        ax.plot(5.15, y, 'ro', markersize=5)

    ax.text(6.5, 6, '5-HT2A', fontsize=8, color='red', fontweight='bold')

    # Labels
    ax.text(3.5, 3, 'Soma', fontsize=8, ha='right')
    ax.text(3.5, 6, 'Apical\ndendrite', fontsize=8, ha='right')
    ax.text(5, 0, 'Output', fontsize=8, ha='center')

    # Arrow showing gain amplification
    ax.annotate('', xy=(8, 6), xytext=(6.5, 6.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
    ax.text(8.2, 6, 'Gain\namplification', fontsize=8, color='red')

    plt.tight_layout()
    plt.savefig('fig1_eigenmode_mechanism.pdf')
    plt.savefig('fig1_eigenmode_mechanism.png')
    print('Saved fig1_eigenmode_mechanism.pdf')
    plt.close()


def fig2_three_phase_model():
    """Figure 2: Three-phase model of psychedelic dimensionality dynamics."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Time axis (arbitrary units representing hours to days)
    t = np.linspace(0, 100, 1000)

    # Create three-phase trajectory
    # Phase 1: Overshoot (0-10 units, rapid rise and peak)
    # Phase 2: Refractory (10-40 units, dip below baseline)
    # Phase 3: Recanalization (40-100, gradual return to baseline)

    d_baseline = 10  # Baseline D_eff

    def three_phase(t):
        d = np.zeros_like(t)
        # Phase 1: Overshoot (Gaussian rise to peak)
        phase1_mask = t < 10
        d[phase1_mask] = d_baseline + 8 * np.exp(-((t[phase1_mask] - 6)**2) / 6)

        # Phase 2: Refractory (exponential decay below baseline)
        phase2_mask = (t >= 10) & (t < 40)
        t2 = t[phase2_mask] - 10
        d[phase2_mask] = d_baseline - 3 * np.exp(-t2/10) + 3 * (1 - np.exp(-t2/10)) * (1 - np.exp(-(t2-5)**2/50))

        # Phase 3: Recanalization (gradual return)
        phase3_mask = t >= 40
        t3 = t[phase3_mask] - 40
        d[phase3_mask] = d_baseline - 1.5 * np.exp(-t3/20)

        return d

    d_eff = three_phase(t)

    # Plot trajectory
    ax.plot(t, d_eff, 'k-', lw=2.5)
    ax.axhline(d_baseline, color='gray', linestyle='--', lw=1, alpha=0.7)

    # Phase regions with shading
    ax.axvspan(0, 10, alpha=0.15, color='red', label='Phase 1: Overshoot')
    ax.axvspan(10, 40, alpha=0.15, color='blue', label='Phase 2: Refractory')
    ax.axvspan(40, 100, alpha=0.15, color='green', label='Phase 3: Recanalization')

    # Annotations
    ax.annotate('Peak\nexpansion', xy=(6, 17.5), xytext=(15, 19),
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
                fontsize=9, color='red', ha='center')

    ax.annotate('Refractory\ndip', xy=(18, 7.5), xytext=(25, 5),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1),
                fontsize=9, color='blue', ha='center')

    ax.annotate('New attractor\nlandscape', xy=(70, 9.5), xytext=(80, 12),
                arrowprops=dict(arrowstyle='->', color='green', lw=1),
                fontsize=9, color='darkgreen', ha='center')

    # Labels
    ax.set_xlabel('Time (hours → days)', fontsize=11)
    ax.set_ylabel('Effective Dimensionality ($D_{eff}$)', fontsize=11)
    ax.set_title('Three-Phase Model of Psychedelic Dimensionality Dynamics', fontsize=12, fontweight='bold')

    # Custom x-axis labels
    ax.set_xticks([0, 5, 10, 25, 40, 70, 100])
    ax.set_xticklabels(['0h', '2h', '6h', '24h', '3d', '7d', '14d'])

    ax.set_ylim(4, 22)
    ax.set_xlim(-2, 102)

    # Add baseline label
    ax.text(95, d_baseline + 0.5, 'Baseline', fontsize=9, color='gray', ha='right')

    ax.legend(loc='upper right', frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig('fig2_three_phase_model.pdf')
    plt.savefig('fig2_three_phase_model.png')
    print('Saved fig2_three_phase_model.pdf')
    plt.close()


def fig3_meg_compound_comparison():
    """Figure 3: MEG compound comparison - key clinical figure."""

    # Load MEG data
    df = pd.read_csv('../data/meg_deff_results.csv')

    # Use windowed mean for more stable estimates
    df['d_eff'] = df['d_eff_mean_windowed']

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Panel A: Bar plot with error bars showing compound comparison
    ax = axes[0]

    compounds = ['Psilocybin', 'LSD', 'Ketamine', 'Tiagabine']
    results = {}

    for compound in compounds:
        drug_data = df[(df['compound_name'] == compound) & (df['condition'] == 'drug')]
        placebo_data = df[(df['compound_name'] == compound) & (df['condition'] == 'placebo')]

        # Get paired subjects
        drug_subjects = set(drug_data['session_id'].values)
        placebo_subjects = set(placebo_data['session_id'].values)
        paired_subjects = list(drug_subjects & placebo_subjects)

        drug_vals = []
        placebo_vals = []
        for subj in paired_subjects:
            drug_val = drug_data[drug_data['session_id'] == subj]['d_eff'].values
            placebo_val = placebo_data[placebo_data['session_id'] == subj]['d_eff'].values
            if len(drug_val) > 0 and len(placebo_val) > 0:
                drug_vals.append(drug_val[0])
                placebo_vals.append(placebo_val[0])

        drug_vals = np.array(drug_vals)
        placebo_vals = np.array(placebo_vals)

        if len(drug_vals) > 1:
            change = (drug_vals.mean() - placebo_vals.mean()) / placebo_vals.mean() * 100
            t_stat, p_val = stats.ttest_rel(drug_vals, placebo_vals)
            d_cohen = (drug_vals.mean() - placebo_vals.mean()) / np.std(drug_vals - placebo_vals)
            se_change = np.std((drug_vals - placebo_vals) / placebo_vals * 100) / np.sqrt(len(drug_vals))
            results[compound] = {
                'change': change,
                'se': se_change,
                'p': p_val,
                'd': d_cohen,
                'n': len(drug_vals)
            }

    # Plot bars
    x = np.arange(len(compounds))
    changes = [results[c]['change'] for c in compounds]
    errors = [results[c]['se'] for c in compounds]
    colors = [COLORS[c] for c in compounds]

    bars = ax.bar(x, changes, yerr=errors, capsize=5, color=colors, edgecolor='black', lw=1)

    # Add significance stars
    for i, compound in enumerate(compounds):
        p = results[compound]['p']
        y = changes[i]
        y_offset = errors[i] + 1.5 if y > 0 else errors[i] - 3
        if p < 0.001:
            ax.text(i, y + y_offset, '***', ha='center', fontsize=12, fontweight='bold')
        elif p < 0.01:
            ax.text(i, y + y_offset, '**', ha='center', fontsize=12, fontweight='bold')
        elif p < 0.1:
            ax.text(i, y + y_offset, '*', ha='center', fontsize=10)

    ax.axhline(0, color='black', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(compounds)
    ax.set_ylabel('Change in Oscillatory Coherence (%)')
    ax.set_title('A. MEG Compound Comparison', fontweight='bold')
    ax.set_ylim(-25, 20)

    # Add mechanism labels
    ax.text(0.5, -22, '5-HT2A agonists', ha='center', fontsize=9, fontstyle='italic')
    ax.text(2, 16, 'NMDA antagonist', ha='center', fontsize=9, fontstyle='italic')
    ax.text(3, 16, 'GABA modulator', ha='center', fontsize=9, fontstyle='italic')

    # Panel B: Individual subject plot for psilocybin
    ax = axes[1]

    # Get psilocybin paired data
    psi_drug = df[(df['compound_name'] == 'Psilocybin') & (df['condition'] == 'drug')]
    psi_placebo = df[(df['compound_name'] == 'Psilocybin') & (df['condition'] == 'placebo')]

    drug_subjects = set(psi_drug['session_id'].values)
    placebo_subjects = set(psi_placebo['session_id'].values)
    paired_subjects = sorted(list(drug_subjects & placebo_subjects))

    for i, subj in enumerate(paired_subjects):
        placebo_val = psi_placebo[psi_placebo['session_id'] == subj]['d_eff'].values[0]
        drug_val = psi_drug[psi_drug['session_id'] == subj]['d_eff'].values[0]

        color = 'indianred' if drug_val < placebo_val else 'steelblue'
        ax.plot([0, 1], [placebo_val, drug_val], 'o-', color=color, alpha=0.5, markersize=6)

    # Add means
    placebo_mean = psi_placebo[psi_placebo['session_id'].isin(paired_subjects)]['d_eff'].mean()
    drug_mean = psi_drug[psi_drug['session_id'].isin(paired_subjects)]['d_eff'].mean()
    ax.plot([0, 1], [placebo_mean, drug_mean], 'ko-', lw=3, markersize=10, label='Group mean')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Placebo', 'Psilocybin'])
    ax.set_ylabel('Oscillatory Coherence (participation ratio)')
    ax.set_title('B. Psilocybin: Individual Trajectories', fontweight='bold')
    ax.legend(loc='upper right', frameon=False)

    # Add stats annotation
    ax.text(0.5, 10.5, f'p = 0.003**\nCohen\'s d = -0.78', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('fig3_meg_compound_comparison.pdf')
    plt.savefig('fig3_meg_compound_comparison.png')
    print('Saved fig3_meg_compound_comparison.pdf')
    plt.close()


def fig4_mechanism_specificity():
    """Figure 4: Mechanism specificity - psychedelics vs ketamine."""

    fig, ax = plt.subplots(figsize=(8, 6))

    # Data from our analysis
    compounds = ['Psilocybin\n(5-HT2A)', 'LSD\n(5-HT2A)', 'Ketamine\n(NMDA)', 'Tiagabine\n(GABA)']
    changes = [-15.0, -13.4, +5.7, +10.8]
    p_vals = [0.003, 0.082, 0.290, 0.307]
    d_vals = [-0.78, -0.50, +0.26, +0.28]

    # Color by mechanism
    colors = ['#7B68EE', '#FF6B6B', '#4ECDC4', '#95A5A6']

    x = np.arange(len(compounds))
    bars = ax.bar(x, changes, color=colors, edgecolor='black', lw=1.5, width=0.6)

    # Add significance indicators and effect sizes
    for i, (p, d, change) in enumerate(zip(p_vals, d_vals, changes)):
        y_text = change + 2 if change > 0 else change - 3

        if p < 0.01:
            sig = '**'
        elif p < 0.1:
            sig = '*'
        else:
            sig = 'ns'

        ax.text(i, y_text, f'{sig}\nd={d:.2f}', ha='center', fontsize=9, fontweight='bold')

    ax.axhline(0, color='black', lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(compounds, fontsize=10)
    ax.set_ylabel('Change in MEG Coherence (%)', fontsize=11)
    ax.set_title('Mechanism Specificity: 5-HT2A Agonists Desynchronize,\nNMDA Antagonists Do Not',
                 fontsize=12, fontweight='bold')

    ax.set_ylim(-25, 20)

    # Add bracket for psychedelics
    ax.annotate('', xy=(-0.3, -20), xytext=(1.3, -20),
                arrowprops=dict(arrowstyle='-', color='purple', lw=2))
    ax.text(0.5, -22, 'Classical psychedelics\n(desynchronization)', ha='center', fontsize=9,
            color='purple', fontweight='bold')

    # Add annotation for clinical implication
    ax.text(2.5, 15, 'No desynchronization\n(different mechanism)', ha='center', fontsize=9,
            color='#4ECDC4', fontstyle='italic')

    plt.tight_layout()
    plt.savefig('fig4_mechanism_specificity.pdf')
    plt.savefig('fig4_mechanism_specificity.png')
    print('Saved fig4_mechanism_specificity.pdf')
    plt.close()


if __name__ == '__main__':
    print('Generating publication figures...')
    fig1_eigenmode_mechanism()
    fig2_three_phase_model()
    fig3_meg_compound_comparison()
    fig4_mechanism_specificity()
    print('Done!')
