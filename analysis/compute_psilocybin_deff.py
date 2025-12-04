#!/usr/bin/env python3
"""
Compute effective dimensionality (D_eff) from psilocybin precision functional mapping data.

Dataset: ds006072 (Siegel et al., 2025)
- 7 subjects (P1-P7), cross-over design
- Psilocybin 25mg vs Methylphenidate 40mg (active control)
- Dense baseline imaging + drug session + longitudinal follow-up

This script compares D_eff between:
1. Baseline sessions (pre-drug)
2. Acute psilocybin session (60-90 min post-dose)

Reference:
    Siegel, J.S. et al. (2025). Psilocybin's acute and persistent brain effects:
    a precision imaging drug trial. Scientific Data.
"""

import numpy as np
import os
import glob
import pandas as pd
from pathlib import Path
from compute_dimensionality import participation_ratio, load_fmri_timeseries

# Subject IDs in the dataset
SUBJECTS = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']

# Based on behavioral data: dose_1/dose_2 with MEQ scores help identify psilocybin
# High MEQ (>60) = psilocybin, Low MEQ (<15) = methylphenidate
PSILOCYBIN_DOSE_SESSION = {
    'P1': 'dose_1',  # MEQ ~3 (low) on dose_1 -> dose_1 is MTP, dose_2 is PSIL
    'P2': 'dose_1',  # MEQ ~150 (high) on dose_1 -> dose_1 is PSIL
    # ... to be determined from MEQ data
}


def discover_sessions(data_dir):
    """Discover available sessions for each subject."""
    session_info = {}

    for sub in SUBJECTS:
        sub_dir = os.path.join(data_dir, f'sub-{sub}')
        if not os.path.exists(sub_dir):
            continue

        sessions = sorted(glob.glob(os.path.join(sub_dir, 'ses-*')))
        session_names = [os.path.basename(s) for s in sessions]
        session_info[sub] = session_names
        print(f"  {sub}: {len(session_names)} sessions: {session_names[:3]}...")

    return session_info


def classify_sessions_by_type(session_names):
    """
    Classify sessions into baseline, drug, and follow-up categories.

    The dataset uses numeric session IDs (ses-0, ses-1, etc.)
    - Early sessions (0-8): baseline
    - Middle sessions (around 9-10): drug administration
    - Later sessions: follow-up

    We'll need to consult the session notes file for exact mapping.
    """
    baseline = []
    drug = []
    followup = []

    for ses in session_names:
        # Extract session number
        try:
            num = int(ses.replace('ses-', ''))
        except ValueError:
            continue

        # Rough classification based on precision mapping protocol
        # Typically: ~10 baseline sessions, then dose day, then follow-up
        if num <= 8:
            baseline.append(ses)
        elif num in [9, 10, 23, 24, 36, 37]:  # Potential drug sessions
            drug.append(ses)
        else:
            followup.append(ses)

    return {
        'baseline': baseline,
        'drug': drug,
        'followup': followup
    }


def analyze_psilocybin_dataset(data_dir):
    """
    Analyze the psilocybin precision functional mapping dataset.

    Expected BIDS structure:
    data_dir/
        sub-P1/
            ses-0/
                func/sub-P1_ses-0_task-rest_bold.nii.gz
            ses-1/
            ...
        sub-P2/
        ...
    """
    results = {
        'subject': [],
        'session': [],
        'session_type': [],
        'd_eff': [],
        'n_timepoints': []
    }

    print("\nDiscovering sessions...")
    session_info = discover_sessions(data_dir)

    if not session_info:
        print("No subjects found in data directory!")
        return None

    for sub, sessions in session_info.items():
        print(f"\nProcessing {sub}...")
        sub_dir = os.path.join(data_dir, f'sub-{sub}')

        # Classify sessions
        session_types = classify_sessions_by_type(sessions)

        # Process a few baseline sessions and any drug sessions
        sessions_to_process = []

        # Take first 3 baseline sessions for average
        for ses in session_types['baseline'][:3]:
            sessions_to_process.append((ses, 'baseline'))

        # Take drug sessions (if identifiable)
        for ses in session_types['drug'][:2]:
            sessions_to_process.append((ses, 'acute'))

        for ses_name, ses_type in sessions_to_process:
            # Find functional file
            func_pattern = os.path.join(
                sub_dir, ses_name, 'func',
                f'sub-{sub}_{ses_name}_task-rest*_bold.nii.gz'
            )
            func_files = sorted(glob.glob(func_pattern))

            if not func_files:
                # Try alternative pattern
                func_pattern = os.path.join(
                    sub_dir, ses_name, 'func',
                    f'sub-{sub}_{ses_name}_*rest*.nii.gz'
                )
                func_files = sorted(glob.glob(func_pattern))

            if not func_files:
                print(f"  No rest file found for {ses_name}")
                continue

            func_file = func_files[0]
            print(f"  Processing {ses_name} ({ses_type}): {os.path.basename(func_file)}")

            try:
                # Extract time series using Schaefer atlas
                ts = load_fmri_timeseries(func_file)

                # Compute D_eff
                d_eff, eigenvalues = participation_ratio(ts)

                results['subject'].append(sub)
                results['session'].append(ses_name)
                results['session_type'].append(ses_type)
                results['d_eff'].append(d_eff)
                results['n_timepoints'].append(ts.shape[0])

                print(f"    D_eff = {d_eff:.2f} (n={ts.shape[0]} TRs)")

            except Exception as e:
                print(f"    Error: {e}")

    return results


def main():
    """Main analysis pipeline."""
    data_dir = "/Users/iantodd/Desktop/highdimensional/25_lsd_dimensionality/data/ds006072"

    print("="*60)
    print("Psilocybin Precision Functional Mapping - D_eff Analysis")
    print("="*60)

    if not os.path.exists(data_dir):
        print(f"\nData directory not found: {data_dir}")
        print("Please download the dataset first.")
        return

    # Check for subject folders
    subject_dirs = glob.glob(os.path.join(data_dir, 'sub-*'))
    if not subject_dirs:
        print("\nNo subject folders found yet. Download may still be in progress.")
        print(f"Data directory: {data_dir}")
        return

    print(f"\nFound {len(subject_dirs)} subject folders")

    # Run analysis
    results = analyze_psilocybin_dataset(data_dir)

    if results and len(results['subject']) > 0:
        # Create DataFrame
        df = pd.DataFrame(results)

        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)

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

                # Pooled std for Cohen's d
                pooled_std = np.sqrt(
                    (baseline_deff.std()**2 + acute_deff.std()**2) / 2
                )
                cohens_d = (mean_acute - mean_baseline) / pooled_std if pooled_std > 0 else 0

                print(f"\nBaseline D_eff: {mean_baseline:.2f} +/- {baseline_deff.std():.2f}")
                print(f"Acute D_eff:    {mean_acute:.2f} +/- {acute_deff.std():.2f}")
                print(f"Change: {pct_change:+.1f}%")
                print(f"Cohen's d: {cohens_d:.2f}")

        # Save results
        output_file = os.path.join(os.path.dirname(data_dir), 'psilocybin_deff_results.csv')
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        return df
    else:
        print("\nNo results obtained. Check data availability.")
        return None


if __name__ == "__main__":
    main()
