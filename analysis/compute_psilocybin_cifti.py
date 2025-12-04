#!/usr/bin/env python3
"""
Compute effective dimensionality (D_eff) from psilocybin CIFTI preprocessed data.

Dataset: ds006072 (Siegel et al., 2025)
Uses preprocessed CIFTI dense time series from NON_BIDS/ciftis/

Session types:
- Baseline1-5: Pre-drug baseline sessions
- Drug1, Drug2: Two drug sessions (one psilocybin, one methylphenidate)
- Between1-4: Between drug sessions
- After1-8: Post-drug follow-up
- replic_*: Replication protocol sessions

Reference:
    Siegel, J.S. et al. (2025). Psilocybin's acute and persistent brain effects:
    a precision imaging drug trial. Scientific Data.
"""

import numpy as np
import os
import glob
import pandas as pd
from pathlib import Path

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not installed. Install with: pip install nibabel")


def participation_ratio(data):
    """
    Compute participation ratio (effective dimensionality) from time series data.

    Parameters
    ----------
    data : ndarray, shape (n_timepoints, n_features)
        Time series data

    Returns
    -------
    d_eff : float
        Effective dimensionality (participation ratio)
    eigenvalues : ndarray
        Eigenvalues of covariance matrix (sorted descending)
    """
    # Center the data
    data_centered = data - data.mean(axis=0)

    # Compute covariance matrix
    cov = np.cov(data_centered.T)

    # Get eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending

    # Only use positive eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Participation ratio
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues**2)

    d_eff = (sum_lambda**2) / sum_lambda_sq

    return d_eff, eigenvalues


def load_cifti_timeseries(cifti_file):
    """
    Load CIFTI dense time series file and extract grayordinates.

    Parameters
    ----------
    cifti_file : str
        Path to .dtseries.nii file

    Returns
    -------
    data : ndarray, shape (n_timepoints, n_grayordinates)
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required for CIFTI reading")

    # Load CIFTI file
    img = nib.load(cifti_file)
    data = img.get_fdata()

    # CIFTI dtseries has shape (n_timepoints, n_grayordinates)
    print(f"    Loaded CIFTI: shape={data.shape}")

    return data


def parse_session_type(filename):
    """Parse session type from filename."""
    basename = os.path.basename(filename)

    if 'Baseline' in basename:
        return 'baseline'
    elif 'Drug' in basename:
        return 'acute'
    elif 'After' in basename:
        return 'followup'
    elif 'Between' in basename:
        return 'between'
    else:
        return 'other'


def get_session_number(filename):
    """Extract session number from filename."""
    import re
    basename = os.path.basename(filename)

    # Match patterns like Baseline1, Drug2, After3
    match = re.search(r'(Baseline|Drug|After|Between)(\d+)', basename)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return 'unknown'


def analyze_psilocybin_cifti(data_dir):
    """
    Analyze preprocessed CIFTI data from psilocybin dataset.
    """
    cifti_dir = os.path.join(data_dir, 'NON_BIDS', 'ciftis')

    results = {
        'subject': [],
        'session': [],
        'session_type': [],
        'd_eff': [],
        'n_timepoints': [],
        'n_grayordinates': []
    }

    # Find all rsfMRI CIFTI files (preprocessed rs-fMRI)
    cifti_pattern = os.path.join(cifti_dir, '*_rsfMRI_*.dtseries.nii')
    cifti_files = sorted(glob.glob(cifti_pattern))

    if not cifti_files:
        print(f"No CIFTI files found in {cifti_dir}")
        return None

    print(f"Found {len(cifti_files)} CIFTI files")

    # Group by subject
    subjects = set()
    for f in cifti_files:
        basename = os.path.basename(f)
        # Extract subject ID (sub-1, sub-2, etc.)
        sub_id = basename.split('_')[0]
        subjects.add(sub_id)

    print(f"Subjects: {sorted(subjects)}")

    for sub_id in sorted(subjects):
        # Skip replication for main analysis
        if 'replic' in sub_id:
            continue

        print(f"\nProcessing {sub_id}...")

        # Get files for this subject
        sub_files = [f for f in cifti_files if os.path.basename(f).startswith(f"{sub_id}_")]

        # Separate by session type
        baseline_files = [f for f in sub_files if 'Baseline' in os.path.basename(f)]
        drug_files = [f for f in sub_files if 'Drug' in os.path.basename(f)]

        # Process baseline sessions (take first 3)
        for cifti_file in baseline_files[:3]:
            session = get_session_number(cifti_file)
            print(f"  Processing {session}...")

            try:
                data = load_cifti_timeseries(cifti_file)

                # Transpose if needed (should be timepoints x grayordinates)
                if data.shape[0] > data.shape[1]:
                    data = data.T

                # Subsample grayordinates for computational efficiency
                # CIFTI has ~90k grayordinates, we'll use 5000 for D_eff
                n_sample = min(5000, data.shape[1])
                np.random.seed(42)
                idx = np.random.choice(data.shape[1], n_sample, replace=False)
                data_sample = data[:, idx]

                d_eff, _ = participation_ratio(data_sample)

                results['subject'].append(sub_id)
                results['session'].append(session)
                results['session_type'].append('baseline')
                results['d_eff'].append(d_eff)
                results['n_timepoints'].append(data.shape[0])
                results['n_grayordinates'].append(data.shape[1])

                print(f"    D_eff = {d_eff:.2f} (n={data.shape[0]} TRs)")

            except Exception as e:
                print(f"    Error: {e}")

        # Process drug sessions
        for cifti_file in drug_files:
            session = get_session_number(cifti_file)
            print(f"  Processing {session}...")

            try:
                data = load_cifti_timeseries(cifti_file)

                if data.shape[0] > data.shape[1]:
                    data = data.T

                n_sample = min(5000, data.shape[1])
                np.random.seed(42)
                idx = np.random.choice(data.shape[1], n_sample, replace=False)
                data_sample = data[:, idx]

                d_eff, _ = participation_ratio(data_sample)

                results['subject'].append(sub_id)
                results['session'].append(session)
                results['session_type'].append('acute')
                results['d_eff'].append(d_eff)
                results['n_timepoints'].append(data.shape[0])
                results['n_grayordinates'].append(data.shape[1])

                print(f"    D_eff = {d_eff:.2f} (n={data.shape[0]} TRs)")

            except Exception as e:
                print(f"    Error: {e}")

    return results


def main():
    """Main analysis pipeline."""
    data_dir = "/Users/iantodd/Desktop/highdimensional/25_lsd_dimensionality/data/ds006072"

    print("="*60)
    print("Psilocybin Precision Functional Mapping - D_eff Analysis")
    print("Using preprocessed CIFTI data")
    print("="*60)

    if not os.path.exists(data_dir):
        print(f"\nData directory not found: {data_dir}")
        return

    results = analyze_psilocybin_cifti(data_dir)

    if results and len(results['subject']) > 0:
        df = pd.DataFrame(results)

        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)

        print("\nAll results:")
        print(df.to_string(index=False))

        # Summary by session type
        summary = df.groupby('session_type')['d_eff'].agg(['mean', 'std', 'count'])
        print("\nD_eff by Session Type:")
        print(summary)

        # Effect size if we have both conditions
        if 'baseline' in df['session_type'].values and 'acute' in df['session_type'].values:
            baseline_deff = df[df['session_type'] == 'baseline']['d_eff']
            acute_deff = df[df['session_type'] == 'acute']['d_eff']

            if len(baseline_deff) > 0 and len(acute_deff) > 0:
                mean_baseline = baseline_deff.mean()
                mean_acute = acute_deff.mean()
                pct_change = 100 * (mean_acute - mean_baseline) / mean_baseline

                pooled_std = np.sqrt(
                    (baseline_deff.std()**2 + acute_deff.std()**2) / 2
                )
                cohens_d = (mean_acute - mean_baseline) / pooled_std if pooled_std > 0 else 0

                print(f"\n{'='*40}")
                print("EFFECT SIZE (Acute vs Baseline)")
                print(f"{'='*40}")
                print(f"Baseline D_eff: {mean_baseline:.2f} +/- {baseline_deff.std():.2f}")
                print(f"Acute D_eff:    {mean_acute:.2f} +/- {acute_deff.std():.2f}")
                print(f"Change: {pct_change:+.1f}%")
                print(f"Cohen's d: {cohens_d:.2f}")

        # Save results
        output_file = os.path.join(os.path.dirname(data_dir), 'psilocybin_cifti_deff_results.csv')
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        return df
    else:
        print("\nNo results obtained.")
        return None


if __name__ == "__main__":
    main()
