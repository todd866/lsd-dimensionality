#!/usr/bin/env python3
"""
Compute the Spectral Centroid of the cortical eigenmode spectrum.

Hypothesis Test: "Geometric vs. Organic"
- LSD (Geometric) should have a LOWER spectral centroid (energy trapped in structural modes).
- Psilocybin (Organic) should have a HIGHER spectral centroid (energy spilling into high-frequency chaos).

Mathematical Definition:
    Centroid = sum(i * Power_i) / sum(Power_i)
    Where 'i' is the eigenmode index (frequency) and Power_i is the variance explained by that mode.
"""

import numpy as np
import os
import glob
import pandas as pd

try:
    from nilearn.maskers import NiftiLabelsMasker
    from nilearn import datasets
    HAS_NILEARN = True
except ImportError:
    HAS_NILEARN = False


def compute_spectrum_metrics(data):
    """
    Compute spectral centroid and entropy of the eigenmode spectrum.

    Parameters
    ----------
    data : ndarray, shape (n_timepoints, n_features)
        Time series data.

    Returns
    -------
    metrics : dict
        'centroid': The weighted average eigenmode index.
        'entropy': Shannon entropy of the spectrum (flatness).
        'eigenvalues': The full eigenvalue spectrum.
    """
    # 1. Center data
    data_centered = data - data.mean(axis=0)

    # 2. Compute Covariance
    cov = np.cov(data_centered.T)

    # 3. Get Eigenvalues (Power Spectrum)
    eigenvalues = np.linalg.eigvalsh(cov)

    # Sort descending (Index 0 = Highest Variance, Index N = Lowest)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Filter small negatives from numerical error
    eigenvalues = eigenvalues[eigenvalues > 0]

    # Normalize to get probability distribution P
    P = eigenvalues / np.sum(eigenvalues)

    # 4. Compute Spectral Centroid
    # Which "mode index" is the center of mass?
    indices = np.arange(len(P))
    centroid = np.sum(indices * P)

    # 5. Compute Spectral Entropy (flatness measure)
    # Higher entropy = flatter spectrum = more "organic"/white noise
    entropy = -np.sum(P * np.log(P + 1e-12))

    # 6. Compute normalized centroid (0-1 scale)
    centroid_normalized = centroid / len(P)

    return {
        'centroid': centroid,
        'centroid_normalized': centroid_normalized,
        'entropy': entropy,
        'eigenvalues': eigenvalues,
        'n_modes': len(eigenvalues)
    }


def load_fmri_timeseries(nifti_file, atlas='schaefer'):
    """
    Extract regional time series from fMRI using Schaefer atlas.
    """
    if not HAS_NILEARN:
        raise ImportError("nilearn required")

    # Get Schaefer atlas (200 parcels)
    atlas_data = datasets.fetch_atlas_schaefer_2018(n_rois=200, resolution_mm=2)
    atlas_file = atlas_data.maps

    # Create masker
    masker = NiftiLabelsMasker(
        labels_img=atlas_file,
        standardize=True,
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=2.0
    )

    # Extract time series
    time_series = masker.fit_transform(nifti_file)

    return time_series


def analyze_lsd_spectrum(data_dir):
    """
    Analyze spectral properties of LSD vs placebo conditions.

    Returns DataFrame with spectral metrics for each scan.
    """
    results = {
        'subject': [],
        'condition': [],
        'centroid': [],
        'centroid_normalized': [],
        'entropy': [],
        'n_modes': []
    }

    # Find all functional files
    func_pattern = os.path.join(data_dir, 'sub-*/ses-*/func/*_bold.nii.gz')
    func_files = sorted(glob.glob(func_pattern))

    if not func_files:
        print(f"No functional files found in {data_dir}")
        return None

    print(f"Found {len(func_files)} functional files")

    for func_file in func_files:
        # Parse subject and session
        basename = os.path.basename(func_file)
        parts = basename.split('_')
        sub_id = parts[0]
        ses_id = parts[1]

        # Determine condition from session
        if 'LSD' in ses_id or 'lsd' in ses_id.lower():
            condition = 'LSD'
        elif 'PLA' in ses_id or 'placebo' in ses_id.lower():
            condition = 'Placebo'
        else:
            # Try to infer from session number
            condition = 'LSD' if '02' in ses_id else 'Placebo'

        print(f"Processing {sub_id} {ses_id} ({condition})...")

        try:
            # Load time series
            ts = load_fmri_timeseries(func_file)

            # Compute spectral metrics
            metrics = compute_spectrum_metrics(ts)

            results['subject'].append(sub_id)
            results['condition'].append(condition)
            results['centroid'].append(metrics['centroid'])
            results['centroid_normalized'].append(metrics['centroid_normalized'])
            results['entropy'].append(metrics['entropy'])
            results['n_modes'].append(metrics['n_modes'])

            print(f"  Centroid: {metrics['centroid']:.2f}, Entropy: {metrics['entropy']:.2f}")

        except Exception as e:
            print(f"  Error: {e}")

    return pd.DataFrame(results)


def demo_geometric_vs_organic():
    """
    Demonstrate the difference between Geometric (LSD) and Organic (Psilocybin)
    using synthetic spectra to illustrate the hypothesis.
    """
    print("="*60)
    print("HYPOTHESIS TEST: Geometric (LSD) vs Organic (Psilocybin)")
    print("="*60)

    n_modes = 200
    indices = np.arange(n_modes)

    # 1. GEOMETRIC PROFILE (LSD hypothesis)
    # Energy decays rapidly (Power Law). Structure is preserved.
    lsd_spectrum = 1 / (indices + 1)**2.0
    lsd_P = lsd_spectrum / np.sum(lsd_spectrum)
    lsd_centroid = np.sum(indices * lsd_P)
    lsd_entropy = -np.sum(lsd_P * np.log(lsd_P + 1e-12))

    # 2. ORGANIC PROFILE (Psilocybin hypothesis)
    # Energy decays slowly. Chaos/Noise is recruited.
    psil_spectrum = 1 / (indices + 1)**1.2
    psil_P = psil_spectrum / np.sum(psil_spectrum)
    psil_centroid = np.sum(indices * psil_P)
    psil_entropy = -np.sum(psil_P * np.log(psil_P + 1e-12))

    print(f"\n[LSD Model] Geometric/Structured:")
    print(f"  - Energy concentrated in low-order modes.")
    print(f"  - Spectral Centroid: {lsd_centroid:.2f}")
    print(f"  - Spectral Entropy: {lsd_entropy:.2f}")

    print(f"\n[Psilocybin Model] Organic/Chaotic:")
    print(f"  - Energy spills into high-order 'tail' modes.")
    print(f"  - Spectral Centroid: {psil_centroid:.2f}")
    print(f"  - Spectral Entropy: {psil_entropy:.2f}")

    centroid_diff = (psil_centroid - lsd_centroid) / lsd_centroid * 100
    entropy_diff = (psil_entropy - lsd_entropy) / lsd_entropy * 100

    print(f"\nPREDICTED DIFFERENCES:")
    print(f"  Centroid shift: +{centroid_diff:.1f}% toward high-freq")
    print(f"  Entropy shift: +{entropy_diff:.1f}% (flatter spectrum)")

    return {
        'lsd': {'centroid': lsd_centroid, 'entropy': lsd_entropy, 'spectrum': lsd_spectrum},
        'psil': {'centroid': psil_centroid, 'entropy': psil_entropy, 'spectrum': psil_spectrum}
    }


def main():
    """Main analysis."""
    print("="*60)
    print("Spectral Centroid Analysis: Geometric vs Organic Hypothesis")
    print("="*60)

    # First show the theoretical prediction
    print("\n--- THEORETICAL PREDICTION ---")
    demo_geometric_vs_organic()

    # Then analyze real data if available
    data_dir = "/Users/iantodd/Desktop/highdimensional/25_lsd_dimensionality/data/ds003059"

    if os.path.exists(data_dir):
        print("\n--- EMPIRICAL ANALYSIS (LSD Dataset) ---")
        df = analyze_lsd_spectrum(data_dir)

        if df is not None and len(df) > 0:
            print("\n" + "="*60)
            print("RESULTS SUMMARY")
            print("="*60)

            # Group by condition
            summary = df.groupby('condition').agg({
                'centroid': ['mean', 'std'],
                'entropy': ['mean', 'std']
            })
            print("\nSpectral Metrics by Condition:")
            print(summary)

            # Statistical comparison
            if 'LSD' in df['condition'].values and 'Placebo' in df['condition'].values:
                lsd_centroid = df[df['condition'] == 'LSD']['centroid']
                pla_centroid = df[df['condition'] == 'Placebo']['centroid']

                lsd_entropy = df[df['condition'] == 'LSD']['entropy']
                pla_entropy = df[df['condition'] == 'Placebo']['entropy']

                if len(lsd_centroid) > 0 and len(pla_centroid) > 0:
                    centroid_change = 100 * (lsd_centroid.mean() - pla_centroid.mean()) / pla_centroid.mean()
                    entropy_change = 100 * (lsd_entropy.mean() - pla_entropy.mean()) / pla_entropy.mean()

                    print(f"\n{'='*40}")
                    print("LSD vs Placebo:")
                    print(f"{'='*40}")
                    print(f"Centroid change: {centroid_change:+.1f}%")
                    print(f"Entropy change: {entropy_change:+.1f}%")

                    if centroid_change < 0:
                        print("\n>>> RESULT: LSD shows LOWER centroid (more geometric)")
                        print("    Consistent with structured eigenmode hypothesis!")
                    else:
                        print("\n>>> RESULT: LSD shows HIGHER centroid")
                        print("    May indicate different mechanism than expected.")

            # Save results
            output_file = os.path.join(os.path.dirname(data_dir), 'lsd_spectral_centroid_results.csv')
            df.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
    else:
        print(f"\nData directory not found: {data_dir}")


if __name__ == "__main__":
    main()
