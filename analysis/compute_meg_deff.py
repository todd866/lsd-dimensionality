#!/usr/bin/env python3
"""
MEG Dimensionality Analysis (D_eff / Participation Ratio)
==========================================================
Analyzes MEG data from Harvard Dataverse LSD/Ketamine/Psilocybin/Tiagabine dataset.
Computes participation ratio (D_eff) to measure neural dimensionality.

Data: doi:10.7910/DVN/9Q1SKM
Format: FieldTrip .mat files, 271 MEG channels, 1200 Hz, 7 min recordings
"""

import numpy as np
import scipy.io as sio
import os
import glob
from pathlib import Path
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path('/Users/iantodd/Desktop/highdimensional/25_lsd_dimensionality/data/lsd_meg')
OUTPUT_DIR = Path('/Users/iantodd/Desktop/highdimensional/25_lsd_dimensionality/data')

# Compound mappings
COMPOUND_MAP = {
    'KET': 'Ketamine',
    'LSD': 'LSD',
    'PMP': 'Psilocybin',
    'TGB': 'Tiagabine'
}

def participation_ratio(data):
    """
    Compute participation ratio (D_eff) from channel x time data.

    D_eff = (sum λ_i)² / sum(λ_i²)

    where λ_i are eigenvalues of the covariance matrix.
    D_eff ranges from 1 (single dominant component) to N (uniform variance).

    Parameters
    ----------
    data : ndarray
        Shape (n_channels, n_timepoints)

    Returns
    -------
    d_eff : float
        Participation ratio (effective dimensionality)
    """
    # Center the data
    data_centered = data - data.mean(axis=1, keepdims=True)

    # Compute covariance matrix (channels x channels)
    cov = np.cov(data_centered)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]  # Remove numerical noise

    # Participation ratio
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues**2)

    d_eff = (sum_lambda**2) / sum_lambda_sq

    return d_eff


def load_meg_data(filepath):
    """
    Load MEG data from FieldTrip .mat file.

    Returns
    -------
    data : ndarray
        Shape (n_channels, n_timepoints)
    fs : float
        Sampling rate
    """
    mat = sio.loadmat(filepath, struct_as_record=False, squeeze_me=False)

    # FieldTrip structure: mat['data'][0,0] returns mat_struct object
    data_struct = mat['data'][0, 0]

    # Extract trial data via attribute access, then index nested array
    trial_data = data_struct.trial[0, 0]

    # Get sampling rate via attribute access
    fs = float(data_struct.fsample[0, 0])

    return trial_data, fs


def compute_windowed_deff(data, fs, window_sec=30, overlap=0.5):
    """
    Compute D_eff in sliding windows for temporal dynamics.

    Parameters
    ----------
    data : ndarray
        Shape (n_channels, n_timepoints)
    fs : float
        Sampling rate
    window_sec : float
        Window size in seconds
    overlap : float
        Fraction of overlap between windows

    Returns
    -------
    d_effs : list
        D_eff for each window
    times : list
        Center time of each window
    """
    window_samples = int(window_sec * fs)
    step_samples = int(window_samples * (1 - overlap))
    n_samples = data.shape[1]

    d_effs = []
    times = []

    start = 0
    while start + window_samples <= n_samples:
        window_data = data[:, start:start + window_samples]
        d_eff = participation_ratio(window_data)
        d_effs.append(d_eff)
        times.append((start + window_samples/2) / fs)
        start += step_samples

    return d_effs, times


def parse_filename(filename):
    """
    Parse filename to extract compound, condition, date, subject.

    Format: {COMPOUND}_{CONDITION}_{DATE}_{SUBJECT}.mat
    e.g., LSD_LSD_010514_1.mat = LSD compound, LSD condition (drug), date 010514, subject 1
          LSD_PLA_010514_1.mat = LSD compound, PLA condition (placebo)
    """
    parts = Path(filename).stem.split('_')
    compound = parts[0]  # KET, LSD, PMP, TGB
    condition = parts[1]  # Drug code or PLA
    date = parts[2]
    subject = parts[3]

    is_drug = condition != 'PLA'

    return {
        'compound': compound,
        'compound_name': COMPOUND_MAP.get(compound, compound),
        'condition': 'drug' if is_drug else 'placebo',
        'date': date,
        'subject': subject,
        'session_id': f"{compound}_{date}_{subject}"
    }


def main():
    print("=" * 70)
    print("MEG DIMENSIONALITY ANALYSIS (D_eff / Participation Ratio)")
    print("=" * 70)
    print()

    # Find all MEG files
    mat_files = sorted(glob.glob(str(DATA_DIR / '*.mat')))
    print(f"Found {len(mat_files)} MEG files")
    print()

    # Results storage
    results = []

    # Process each file
    for i, filepath in enumerate(mat_files):
        filename = os.path.basename(filepath)
        info = parse_filename(filename)

        print(f"[{i+1:3d}/{len(mat_files)}] {filename} ({info['compound_name']}, {info['condition']})")

        try:
            # Load data
            data, fs = load_meg_data(filepath)
            n_channels, n_timepoints = data.shape
            duration_sec = n_timepoints / fs

            # Compute global D_eff
            d_eff_global = participation_ratio(data)

            # Compute windowed D_eff for temporal dynamics
            d_effs_windowed, times = compute_windowed_deff(data, fs, window_sec=30, overlap=0.5)
            d_eff_mean = np.mean(d_effs_windowed)
            d_eff_std = np.std(d_effs_windowed)

            result = {
                'filename': filename,
                'compound': info['compound'],
                'compound_name': info['compound_name'],
                'condition': info['condition'],
                'subject': info['subject'],
                'session_id': info['session_id'],
                'n_channels': n_channels,
                'duration_sec': duration_sec,
                'fs': fs,
                'd_eff_global': d_eff_global,
                'd_eff_mean_windowed': d_eff_mean,
                'd_eff_std_windowed': d_eff_std,
                'n_windows': len(d_effs_windowed)
            }
            results.append(result)

            print(f"         D_eff = {d_eff_global:.2f} (global), {d_eff_mean:.2f} ± {d_eff_std:.2f} (windowed)")

        except Exception as e:
            print(f"         ERROR: {e}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    output_file = OUTPUT_DIR / 'meg_deff_results.csv'
    df.to_csv(output_file, index=False)
    print()
    print(f"Results saved to: {output_file}")
    print()

    # Statistical analysis
    print("=" * 70)
    print("STATISTICAL ANALYSIS: Drug vs Placebo D_eff")
    print("=" * 70)
    print()

    for compound in ['LSD', 'KET', 'PMP', 'TGB']:
        compound_name = COMPOUND_MAP[compound]
        drug_data = df[(df['compound'] == compound) & (df['condition'] == 'drug')]['d_eff_global']
        placebo_data = df[(df['compound'] == compound) & (df['condition'] == 'placebo')]['d_eff_global']

        if len(drug_data) > 0 and len(placebo_data) > 0:
            # Paired t-test (within-subject design)
            t_stat, p_val = stats.ttest_ind(drug_data, placebo_data)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((drug_data.std()**2 + placebo_data.std()**2) / 2)
            cohens_d = (drug_data.mean() - placebo_data.mean()) / pooled_std if pooled_std > 0 else 0

            # Percent change
            pct_change = ((drug_data.mean() - placebo_data.mean()) / placebo_data.mean()) * 100

            print(f"{compound_name:12s}: Drug={drug_data.mean():.2f}±{drug_data.std():.2f}, "
                  f"Placebo={placebo_data.mean():.2f}±{placebo_data.std():.2f}")
            print(f"              Δ = {pct_change:+.1f}%, Cohen's d = {cohens_d:.2f}, "
                  f"t = {t_stat:.2f}, p = {p_val:.4f}")

            # Significance indicator
            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "n.s."
            print(f"              {sig}")
            print()

    # Summary table
    print("=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    summary = df.groupby(['compound_name', 'condition'])['d_eff_global'].agg(['mean', 'std', 'count'])
    print(summary.round(2))

    return df


if __name__ == '__main__':
    df = main()
