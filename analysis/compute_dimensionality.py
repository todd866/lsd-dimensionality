#!/usr/bin/env python3
"""
Compute effective dimensionality (D_eff) from fMRI data.

This script:
1. Loads preprocessed fMRI time series (from ROI parcellation)
2. Computes covariance matrix across ROIs
3. Calculates participation ratio (D_eff)
4. Compares LSD vs placebo conditions

Effective dimensionality via participation ratio:
    D_eff = (Σ λ_i)² / (Σ λ_i²)

Where λ_i are eigenvalues of the covariance matrix.
"""

import numpy as np
import os
import glob
from pathlib import Path

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not installed. Install with: pip install nibabel")

try:
    from nilearn import datasets, input_data, image
    from nilearn.maskers import NiftiLabelsMasker
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False
    print("Warning: nilearn not installed. Install with: pip install nilearn")

def participation_ratio(data):
    """
    Compute participation ratio (effective dimensionality) from time series data.

    Parameters
    ----------
    data : ndarray, shape (n_timepoints, n_features)
        Time series data (e.g., ROI time series from fMRI)

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

def load_fmri_timeseries(fmri_file, atlas='schaefer'):
    """
    Load fMRI file and extract ROI time series using atlas parcellation.

    Parameters
    ----------
    fmri_file : str
        Path to 4D fMRI NIfTI file
    atlas : str
        Atlas to use ('schaefer', 'aal', or 'harvard_oxford')

    Returns
    -------
    time_series : ndarray, shape (n_timepoints, n_rois)
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn required for ROI extraction")

    # Get atlas
    if atlas == 'schaefer':
        atlas_data = datasets.fetch_atlas_schaefer_2018(n_rois=200)
        atlas_file = atlas_data.maps
    elif atlas == 'aal':
        atlas_data = datasets.fetch_atlas_aal()
        atlas_file = atlas_data.maps
    else:
        atlas_data = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        atlas_file = atlas_data.maps

    # Create masker
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        standardize=True,
        memory='nilearn_cache',
        verbose=0
    )

    # Extract time series
    time_series = masker.fit_transform(fmri_file)

    return time_series

def analyze_lsd_dataset(data_dir):
    """
    Analyze the LSD OpenNeuro dataset (ds003059).

    Expected BIDS structure:
    data_dir/
        sub-01/
            ses-lsd/
                func/sub-01_ses-lsd_task-rest_bold.nii.gz
            ses-placebo/
                func/sub-01_ses-placebo_task-rest_bold.nii.gz
        ...
    """
    results = {
        'subject': [],
        'condition': [],
        'd_eff': [],
        'n_timepoints': []
    }

    subjects = sorted(glob.glob(os.path.join(data_dir, 'sub-*')))

    print(f"Found {len(subjects)} subjects")

    for sub_dir in subjects:
        sub_id = os.path.basename(sub_dir)
        print(f"\nProcessing {sub_id}...")

        for condition, folder in [('LSD', 'LSD'), ('placebo', 'PLCB')]:
            # Find functional files (BIDS naming: ses-LSD or ses-PLCB, with run numbers)
            func_pattern = os.path.join(
                sub_dir, f'ses-{folder}', 'func',
                f'{sub_id}_ses-{folder}_task-rest_run-*_bold.nii.gz'
            )
            func_files = sorted(glob.glob(func_pattern))

            if not func_files:
                print(f"  No file found for {condition}")
                continue

            # Use first run for consistency
            func_file = func_files[0]
            print(f"  Loading {condition}: {os.path.basename(func_file)} ({len(func_files)} runs available)")

            try:
                # Extract time series
                ts = load_fmri_timeseries(func_file)

                # Compute D_eff
                d_eff, eigenvalues = participation_ratio(ts)

                results['subject'].append(sub_id)
                results['condition'].append(condition)
                results['d_eff'].append(d_eff)
                results['n_timepoints'].append(ts.shape[0])

                print(f"    D_eff ({condition}): {d_eff:.2f}")

            except Exception as e:
                print(f"    Error: {e}")

    return results

def demo_synthetic():
    """
    Demo with synthetic data showing D_eff difference between
    'baseline' (correlated) and 'psychedelic' (decorrelated) states.
    """
    print("="*60)
    print("Demo: Synthetic Data")
    print("="*60)

    np.random.seed(42)
    n_timepoints = 500
    n_features = 100

    # Baseline: strongly correlated (low D_eff)
    # Create data dominated by a few principal components
    n_components_baseline = 5
    latent = np.random.randn(n_timepoints, n_components_baseline)
    mixing_baseline = np.random.randn(n_components_baseline, n_features)
    baseline_data = latent @ mixing_baseline + 0.3 * np.random.randn(n_timepoints, n_features)

    # Psychedelic: decorrelated (high D_eff)
    # More independent components, weaker correlations
    n_components_psychedelic = 30
    latent_psych = np.random.randn(n_timepoints, n_components_psychedelic)
    mixing_psych = np.random.randn(n_components_psychedelic, n_features)
    psychedelic_data = latent_psych @ mixing_psych + 0.5 * np.random.randn(n_timepoints, n_features)

    # Compute D_eff
    d_eff_baseline, eig_baseline = participation_ratio(baseline_data)
    d_eff_psychedelic, eig_psych = participation_ratio(psychedelic_data)

    print(f"\nBaseline condition:")
    print(f"  D_eff = {d_eff_baseline:.2f}")
    print(f"  Top 5 eigenvalues: {eig_baseline[:5].round(2)}")

    print(f"\nPsychedelic condition:")
    print(f"  D_eff = {d_eff_psychedelic:.2f}")
    print(f"  Top 5 eigenvalues: {eig_psych[:5].round(2)}")

    print(f"\nD_eff ratio (psychedelic/baseline): {d_eff_psychedelic/d_eff_baseline:.2f}x")

    return d_eff_baseline, d_eff_psychedelic

if __name__ == "__main__":
    # Run demo with synthetic data
    demo_synthetic()

    # If real data is available, analyze it
    data_dir = "/Users/iantodd/Desktop/highdimensional/25_lsd_dimensionality/data/ds003059"

    if os.path.exists(data_dir):
        print("\n" + "="*60)
        print("Analyzing LSD Dataset (ds003059)")
        print("="*60)
        results = analyze_lsd_dataset(data_dir)

        # Summary statistics
        import pandas as pd
        df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(df.groupby('condition')['d_eff'].agg(['mean', 'std', 'count']))
    else:
        print(f"\nReal data not found at {data_dir}")
        print("Run download_lsd_data.py first, or use the demo above.")
