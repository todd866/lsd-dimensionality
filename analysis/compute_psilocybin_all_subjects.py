#!/usr/bin/env python3
"""
Compute effective dimensionality (D_eff) and spectral centroid for ALL subjects
in the Siegel psilocybin precision mapping dataset (ds006072).

This script works with the preprocessed CIFTIs from NON_BIDS/ciftis/ which use
flat naming: sub-{N}_{Session}_rsfMRI_uout_bpss_sr_noGSR_sm4.dtseries.nii

Session types:
- Baseline1-5: Pre-drug baseline sessions
- Drug1-2: During psilocybin (or methylphenidate for control)
- Between1-4: Between drug exposures
- After1-8: Post-drug sessions
- replic_*: Replication sessions (6+ months later, N=4 subjects)

N=7 primary subjects, N=4 with replication data.
"""

import numpy as np
import glob
import os
import re
from pathlib import Path
import pandas as pd
from scipy import stats

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not installed. Install with: pip install nibabel")


def participation_ratio(data):
    """
    Compute participation ratio (effective dimensionality) from time series data.

    Uses SVD on the temporal covariance (T×T) instead of spatial covariance (N×N)
    to avoid memory issues with high-dimensional grayordinate data.

    For data X (T×N), the singular values σ relate to eigenvalues λ of X'X by λ = σ²/T.
    The participation ratio is scale-invariant, so we can use σ² directly.

    Parameters
    ----------
    data : ndarray, shape (n_timepoints, n_features)
        Time series data

    Returns
    -------
    d_eff : float
        Effective dimensionality (participation ratio)
    eigenvalues : ndarray
        Squared singular values (proportional to eigenvalues), sorted descending
    """
    # Center the data
    data_centered = data - data.mean(axis=0)

    # Use SVD for memory efficiency: X = U @ S @ Vt
    # Eigenvalues of X'X are S², but we only need the ratio
    # Computing full SVD on (T×N) where T << N gives us T singular values
    # This is O(T²N) instead of O(N²T) and uses O(T²) memory instead of O(N²)

    # Use truncated SVD - only compute singular values (faster)
    # scipy.linalg.svd with full_matrices=False is memory efficient
    from scipy.linalg import svd

    # Compute SVD of centered data
    # For (T×N) matrix where T << N, this gives T singular values
    try:
        # Use gesdd driver for speed, fall back to gesvd if it fails
        _, s, _ = svd(data_centered, full_matrices=False, lapack_driver='gesdd')
    except:
        _, s, _ = svd(data_centered, full_matrices=False)

    # Squared singular values (proportional to eigenvalues)
    eigenvalues = s**2
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Participation ratio: (sum λ)² / sum(λ²)
    sum_lambda = np.sum(eigenvalues)
    sum_lambda_sq = np.sum(eigenvalues**2)

    d_eff = (sum_lambda**2) / sum_lambda_sq

    return d_eff, eigenvalues


def spectral_centroid(eigenvalues):
    """
    Compute spectral centroid (center of mass of eigenspectrum).

    Higher values indicate energy distributed toward higher modes.
    """
    # Normalize eigenvalues to sum to 1
    eigenvalues = eigenvalues / np.sum(eigenvalues)

    # Compute centroid (weighted average of mode indices)
    indices = np.arange(len(eigenvalues))
    centroid = np.sum(indices * eigenvalues)

    return centroid


def load_cifti_timeseries(cifti_file):
    """
    Load CIFTI dtseries file and extract grayordinate time series.

    Returns
    -------
    data : ndarray, shape (n_timepoints, n_grayordinates)
    """
    if not HAS_NIBABEL:
        raise ImportError("nibabel required")

    img = nib.load(cifti_file)
    data = img.get_fdata()

    # CIFTI data is (n_timepoints, n_grayordinates)
    return data


def parse_cifti_filename(filename):
    """
    Parse preprocessed CIFTI filename to extract subject and session info.

    Filename format: sub-{N}_{Session}_rsfMRI_uout_bpss_sr_noGSR_sm4.dtseries.nii
    Examples:
        sub-1_Baseline1_rsfMRI_uout_bpss_sr_noGSR_sm4.dtseries.nii
        sub-1_Drug1_rsfMRI_uout_bpss_sr_noGSR_sm4.dtseries.nii
        sub-1_replic_Baseline1_rsfMRI_uout_bpss_sr_noGSR_sm4.dtseries.nii

    Returns
    -------
    subject : str
        Subject ID (e.g., 'sub-1')
    session : str
        Session name (e.g., 'Baseline1', 'Drug1', 'replic_Baseline1')
    session_type : str
        One of: 'baseline', 'drug', 'between', 'after', 'unknown'
    is_replication : bool
        True if this is a replication session
    """
    basename = os.path.basename(filename)

    # Match pattern: sub-{N}_{Session}_rsfMRI_...
    match = re.match(r'sub-(\d+)_(.+?)_rsfMRI_', basename)
    if not match:
        # Try alternate pattern with ctx
        match = re.match(r'sub-(\d+)_(.+?)_upck_', basename)

    if not match:
        return None, None, 'unknown', False

    sub_num = match.group(1)
    session_raw = match.group(2)
    subject = f'sub-{sub_num}'

    # Check for replication sessions
    is_replication = session_raw.startswith('replic_')
    session = session_raw.replace('replic_', '') if is_replication else session_raw

    # Classify session type
    session_lower = session.lower()
    if 'baseline' in session_lower:
        session_type = 'baseline'
    elif 'drug' in session_lower:
        session_type = 'drug'
    elif 'between' in session_lower:
        session_type = 'between'
    elif 'after' in session_lower:
        session_type = 'after'
    else:
        session_type = 'unknown'

    return subject, session_raw, session_type, is_replication


def analyze_all_ciftis(data_dir):
    """
    Analyze all preprocessed CIFTIs in the flat directory structure.

    The NON_BIDS/ciftis/ folder contains all files with naming:
    sub-{N}_{Session}_rsfMRI_uout_bpss_sr_noGSR_sm4.dtseries.nii

    Returns list of dicts with results.
    """
    results = []

    # Find all preprocessed rsfMRI CIFTIs (not the raw upck_ files)
    cifti_pattern = os.path.join(data_dir, 'sub-*_*_rsfMRI_uout_bpss_sr_noGSR_sm4.dtseries.nii')
    cifti_files = sorted(glob.glob(cifti_pattern))

    print(f"Found {len(cifti_files)} preprocessed CIFTI files")

    if not cifti_files:
        print(f"No CIFTI files found matching pattern: {cifti_pattern}")
        return results

    # Group by subject for progress reporting
    subjects = set()
    for f in cifti_files:
        subj, _, _, _ = parse_cifti_filename(f)
        if subj:
            subjects.add(subj)
    print(f"Subjects: {sorted(subjects)}")

    for i, cifti_file in enumerate(cifti_files):
        subject, session, session_type, is_replication = parse_cifti_filename(cifti_file)

        if subject is None:
            print(f"  Skipping (couldn't parse): {os.path.basename(cifti_file)}")
            continue

        try:
            print(f"  [{i+1}/{len(cifti_files)}] {subject} {session} ({session_type}{'*' if is_replication else ''})...", end='', flush=True)
            data = load_cifti_timeseries(cifti_file)

            # Compute metrics
            d_eff, eigenvalues = participation_ratio(data)
            sc = spectral_centroid(eigenvalues)

            results.append({
                'subject': subject,
                'session': session,
                'session_type': session_type,
                'is_replication': is_replication,
                'd_eff': d_eff,
                'spectral_centroid': sc,
                'n_timepoints': data.shape[0],
                'n_grayordinates': data.shape[1]
            })

            print(f" D_eff={d_eff:.1f}, SC={sc:.1f}")

        except Exception as e:
            print(f" ERROR: {e}")

    return results


def main():
    # Use the preprocessed CIFTIs directory (symlinked or direct path)
    data_dir = "/Volumes/Research/ds006072_ciftis"

    # Alternative paths to try
    alt_paths = [
        "/Users/iantodd/Desktop/highdimensional/25_lsd_dimensionality/data/ds006072_ciftis",
        "/Users/iantodd/Desktop/highdimensional/25_lsd_dimensionality/data/ds006072/NON_BIDS/ciftis",
    ]

    if not os.path.exists(data_dir):
        for alt in alt_paths:
            if os.path.exists(alt):
                data_dir = alt
                break
        else:
            print(f"Data directory not found: {data_dir}")
            print("Download CIFTIs with:")
            print("  aws s3 sync --no-sign-request s3://openneuro.org/ds006072/NON_BIDS/ciftis/ /Volumes/Research/ds006072_ciftis/")
            return

    print(f"Using data directory: {data_dir}")

    # Analyze all CIFTIs
    all_results = analyze_all_ciftis(data_dir)

    if not all_results:
        print("No results to save")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save results
    output_file = os.path.join(
        "/Users/iantodd/Desktop/highdimensional/25_lsd_dimensionality/data",
        "psilocybin_all_subjects_results.csv"
    )
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Separate primary and replication data
    df_primary = df[~df['is_replication']]
    df_replic = df[df['is_replication']]

    print(f"\nPrimary sessions: {len(df_primary)}")
    print(f"Replication sessions: {len(df_replic)}")

    # Group by subject and session type (primary data only)
    print("\n--- PRIMARY DATA ---")
    summary = df_primary.groupby(['subject', 'session_type']).agg({
        'd_eff': ['mean', 'std', 'count'],
        'spectral_centroid': ['mean', 'std']
    }).round(2)
    print(summary)

    # Paired comparison: drug vs baseline (primary data)
    print("\n" + "="*60)
    print("PAIRED COMPARISON: DRUG vs BASELINE (Primary Data)")
    print("="*60)

    subjects_with_both = []
    for sub in df_primary['subject'].unique():
        sub_data = df_primary[df_primary['subject'] == sub]
        has_baseline = (sub_data['session_type'] == 'baseline').any()
        has_drug = (sub_data['session_type'] == 'drug').any()
        if has_baseline and has_drug:
            subjects_with_both.append(sub)

    print(f"\nSubjects with both baseline and drug: {len(subjects_with_both)}")

    if len(subjects_with_both) >= 2:
        baseline_means = []
        drug_means = []
        baseline_sc = []
        drug_sc = []

        for sub in subjects_with_both:
            sub_data = df_primary[df_primary['subject'] == sub]
            baseline_means.append(sub_data[sub_data['session_type'] == 'baseline']['d_eff'].mean())
            drug_means.append(sub_data[sub_data['session_type'] == 'drug']['d_eff'].mean())
            baseline_sc.append(sub_data[sub_data['session_type'] == 'baseline']['spectral_centroid'].mean())
            drug_sc.append(sub_data[sub_data['session_type'] == 'drug']['spectral_centroid'].mean())

        baseline_means = np.array(baseline_means)
        drug_means = np.array(drug_means)
        baseline_sc = np.array(baseline_sc)
        drug_sc = np.array(drug_sc)

        # D_eff comparison
        d_eff_change = ((drug_means - baseline_means) / baseline_means * 100).mean()
        t_stat, p_val = stats.ttest_rel(drug_means, baseline_means)
        cohen_d = (drug_means.mean() - baseline_means.mean()) / np.std(drug_means - baseline_means)

        print(f"\nD_eff:")
        print(f"  Baseline mean: {baseline_means.mean():.1f} (SD: {baseline_means.std():.1f})")
        print(f"  Drug mean: {drug_means.mean():.1f} (SD: {drug_means.std():.1f})")
        print(f"  Change: {d_eff_change:.1f}%")
        print(f"  Paired t-test: t={t_stat:.2f}, p={p_val:.4f}")
        print(f"  Cohen's d: {cohen_d:.2f}")

        # Spectral centroid comparison
        sc_change = ((drug_sc - baseline_sc) / baseline_sc * 100).mean()
        t_stat_sc, p_val_sc = stats.ttest_rel(drug_sc, baseline_sc)
        cohen_d_sc = (drug_sc.mean() - baseline_sc.mean()) / np.std(drug_sc - baseline_sc)

        print(f"\nSpectral Centroid:")
        print(f"  Baseline mean: {baseline_sc.mean():.1f} (SD: {baseline_sc.std():.1f})")
        print(f"  Drug mean: {drug_sc.mean():.1f} (SD: {drug_sc.std():.1f})")
        print(f"  Change: {sc_change:.1f}%")
        print(f"  Paired t-test: t={t_stat_sc:.2f}, p={p_val_sc:.4f}")
        print(f"  Cohen's d: {cohen_d_sc:.2f}")
    else:
        print("Not enough subjects with paired data for statistical comparison")

    # Replication analysis (if available)
    if len(df_replic) > 0:
        print("\n" + "="*60)
        print("REPLICATION DATA (6+ months later)")
        print("="*60)
        replic_summary = df_replic.groupby(['subject', 'session_type']).agg({
            'd_eff': ['mean', 'std', 'count'],
            'spectral_centroid': ['mean', 'std']
        }).round(2)
        print(replic_summary)


if __name__ == "__main__":
    main()
