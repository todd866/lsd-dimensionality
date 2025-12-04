#!/usr/bin/env python3
"""
Compute spectral centroid on psilocybin CIFTI data.

Tests the "Geometric vs Organic" hypothesis:
- LSD (Geometric): Lower spectral centroid (energy in structural modes)
- Psilocybin (Organic): Higher spectral centroid (energy spilling into high-freq chaos)

Data: ds006072 NON_BIDS/ciftis preprocessed CIFTI files
"""

import numpy as np
import os
import glob
import pandas as pd
from scipy import stats

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


def compute_spectrum_metrics(data):
    """
    Compute spectral centroid and entropy of the eigenmode spectrum.

    Parameters
    ----------
    data : ndarray, shape (n_timepoints, n_features)
        Time series data (CIFTI grayordinates).

    Returns
    -------
    metrics : dict
        'centroid': Weighted average eigenmode index
        'entropy': Shannon entropy of spectrum
        'd_eff': Effective dimensionality (participation ratio)
    """
    # Center data
    data_centered = data - data.mean(axis=0)

    # Subsample grayordinates for computational efficiency
    n_samples = min(5000, data.shape[1])
    np.random.seed(42)
    idx = np.random.choice(data.shape[1], n_samples, replace=False)
    data_sub = data_centered[:, idx]

    # Compute covariance
    cov = np.cov(data_sub.T)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Normalize to probability distribution
    P = eigenvalues / np.sum(eigenvalues)

    # Spectral centroid (center of mass of eigenspectrum)
    indices = np.arange(len(P))
    centroid = np.sum(indices * P)
    centroid_normalized = centroid / len(P)

    # Spectral entropy
    entropy = -np.sum(P * np.log(P + 1e-12))

    # Effective dimensionality (participation ratio)
    d_eff = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)

    return {
        'centroid': centroid,
        'centroid_normalized': centroid_normalized,
        'entropy': entropy,
        'd_eff': d_eff,
        'n_modes': len(eigenvalues)
    }


def parse_cifti_filename(filename):
    """
    Parse ds006072 CIFTI filename to extract subject and condition.

    Examples:
    - sub-1_Baseline1_rsfMRI_uout_bpss_sr_noGSR_sm4.dtseries.nii
    - sub-1_Drug1_rsfMRI_uout_bpss_sr_noGSR_sm4.dtseries.nii
    """
    basename = os.path.basename(filename)
    parts = basename.split('_')

    subject = parts[0]  # sub-1

    # Parse condition from second part
    condition_raw = parts[1]  # Baseline1, Drug1, After1, etc.

    # Extract condition type
    if 'Baseline' in condition_raw:
        condition = 'Baseline'
    elif 'Drug' in condition_raw:
        condition = 'Drug'  # Psilocybin
    elif 'After' in condition_raw:
        condition = 'After'
    elif 'Between' in condition_raw:
        condition = 'Between'
    else:
        condition = 'Unknown'

    # Extract session number
    import re
    session_match = re.search(r'(\d+)$', condition_raw)
    session = int(session_match.group(1)) if session_match else 0

    return {
        'subject': subject,
        'condition': condition,
        'session': session,
        'condition_raw': condition_raw
    }


def analyze_psilocybin_spectrum(data_dir):
    """
    Analyze spectral properties of psilocybin vs baseline.
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required: pip install nibabel")

    results = {
        'subject': [],
        'condition': [],
        'session': [],
        'centroid': [],
        'centroid_normalized': [],
        'entropy': [],
        'd_eff': [],
        'n_modes': []
    }

    # Find all CIFTI files
    cifti_pattern = os.path.join(data_dir, '*_rsfMRI_*.dtseries.nii')
    cifti_files = sorted(glob.glob(cifti_pattern))

    if not cifti_files:
        print(f"No CIFTI files found in {data_dir}")
        return None

    print(f"Found {len(cifti_files)} CIFTI files")
    print("="*60)

    for cifti_file in cifti_files:
        info = parse_cifti_filename(cifti_file)

        # Only analyze Baseline and Drug conditions for comparison
        if info['condition'] not in ['Baseline', 'Drug']:
            continue

        print(f"Processing {info['subject']} {info['condition_raw']}...")

        try:
            # Load CIFTI
            img = nib.load(cifti_file)
            data = img.get_fdata()  # (timepoints, grayordinates)

            # Compute metrics
            metrics = compute_spectrum_metrics(data)

            results['subject'].append(info['subject'])
            results['condition'].append(info['condition'])
            results['session'].append(info['session'])
            results['centroid'].append(metrics['centroid'])
            results['centroid_normalized'].append(metrics['centroid_normalized'])
            results['entropy'].append(metrics['entropy'])
            results['d_eff'].append(metrics['d_eff'])
            results['n_modes'].append(metrics['n_modes'])

            print(f"  Centroid: {metrics['centroid']:.2f}, D_eff: {metrics['d_eff']:.1f}")

        except Exception as e:
            print(f"  Error: {e}")

    return pd.DataFrame(results)


def main():
    """Main analysis."""
    print("="*60)
    print("PSILOCYBIN SPECTRAL CENTROID ANALYSIS")
    print("Testing: Geometric (LSD) vs Organic (Psilocybin) Hypothesis")
    print("="*60)

    data_dir = "/Users/iantodd/Desktop/highdimensional/25_lsd_dimensionality/data/ds006072/NON_BIDS/ciftis"

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Download data first with:")
        print("aws s3 sync --no-sign-request s3://openneuro.org/ds006072/NON_BIDS/ciftis/ data/ds006072/NON_BIDS/ciftis/")
        return

    # Run analysis
    df = analyze_psilocybin_spectrum(data_dir)

    if df is None or len(df) == 0:
        print("No results")
        return

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)

    # Group by condition
    summary = df.groupby('condition').agg({
        'centroid': ['mean', 'std', 'count'],
        'd_eff': ['mean', 'std']
    })
    print("\nSpectral Metrics by Condition:")
    print(summary)

    # Statistical comparison
    if 'Baseline' in df['condition'].values and 'Drug' in df['condition'].values:
        baseline = df[df['condition'] == 'Baseline']
        drug = df[df['condition'] == 'Drug']

        # Centroid comparison
        centroid_baseline = baseline['centroid'].values
        centroid_drug = drug['centroid'].values

        if len(centroid_baseline) > 1 and len(centroid_drug) > 1:
            t_stat, p_val = stats.ttest_ind(centroid_drug, centroid_baseline)
            change = 100 * (centroid_drug.mean() - centroid_baseline.mean()) / centroid_baseline.mean()

            print(f"\n{'='*50}")
            print("PSILOCYBIN vs BASELINE:")
            print(f"{'='*50}")
            print(f"Baseline centroid: {centroid_baseline.mean():.2f} ± {centroid_baseline.std():.2f}")
            print(f"Drug centroid:     {centroid_drug.mean():.2f} ± {centroid_drug.std():.2f}")
            print(f"Change: {change:+.1f}%")
            print(f"t-test: t={t_stat:.2f}, p={p_val:.4f}")

            print("\n" + "="*50)
            print("INTERPRETATION:")
            print("="*50)
            if change > 0:
                print(">>> PSILOCYBIN shows HIGHER spectral centroid")
                print("    Consistent with 'Organic' hypothesis: energy spills")
                print("    into high-frequency chaotic modes")
            else:
                print(">>> PSILOCYBIN shows LOWER spectral centroid")
                print("    Similar to LSD 'Geometric' pattern")

    # Save results
    output_file = os.path.join(os.path.dirname(data_dir), '../psilocybin_spectral_results.csv')
    output_file = os.path.normpath(output_file)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
