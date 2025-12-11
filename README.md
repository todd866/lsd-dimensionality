# Psychedelics as Dimensionality Modulators

Code and analysis for the paper: **"Psychedelics as Dimensionality Modulators: A Unified Framework for Serotonergic Brain State Expansion"**

Target journal: *Nature Communications*

## Overview

This repository accompanies our theoretical framework proposing that classical psychedelics primarily modulate the **effective dimensionality** of cortical dynamics through 5-HT2A-mediated dendritic gain amplification.

### Key Findings

| Dataset | Compound | D_eff Change | Spectral Centroid | p-value |
|---------|----------|--------------|-------------------|---------|
| ds003059 (N=15) | LSD | +8.6% | +10.0% | p=0.0008 |
| ds006072 (N=1) | Psilocybin | +19.2% | +18.6% | p=0.044 |

**Key finding**: Psilocybin shows ~2x the spectral centroid shift of LSD, quantifying the "organic vs geometric" phenomenological distinction.

### Within-Subject Pharmacological Dissociation

In the psilocybin study (ds006072), a single subject (P1) received both psilocybin and methylphenidate:
- **Psilocybin**: +25.2% D_eff, MEQ Mystical = 4.37/5
- **Methylphenidate**: -15.7% D_eff, MEQ Mystical = 0.0/5

## Structure

```
├── lsd_dimensionality.tex        # Full manuscript (LaTeX)
├── lsd_dimensionality.pdf        # Compiled PDF
├── references.bib                # Bibliography
├── cover_letter.tex              # Cover letter
├── figures/                      # Publication figures
│   ├── generate_figures.py       # Main results figures
│   ├── fig1_eigenmode_mechanism.py  # Conceptual framework (Fig 1)
│   ├── fig2_three_phase_model.py    # Three-phase model (Fig 2)
│   └── *.pdf, *.png              # Generated figures
├── analysis/                     # Analysis code
│   ├── compute_dimensionality.py         # D_eff for LSD dataset
│   ├── compute_spectral_centroid.py      # Spectral centroid for LSD
│   ├── compute_psilocybin_deff.py        # D_eff for psilocybin
│   ├── compute_psilocybin_spectral_centroid.py  # Spectral for psilocybin
│   ├── compute_psilocybin_cifti.py       # CIFTI handling
│   └── download_lsd_data.py              # Data download helper
└── data/                         # fMRI data (not tracked, download from OpenNeuro)
```

## Key Concepts

- **Effective Dimensionality (D_eff)**: Participation ratio of covariance eigenvalues
- **Spectral Centroid**: Center of mass of eigenspectrum (higher = more geometric modes)
- **Three-Phase Model**: Acute expansion → Refractory compression → Recanalization
- **Mechanism**: 5-HT2A activation → dendritic gain amplification → eigenmode expansion

## Running the Analysis

### Install dependencies

```bash
pip install numpy matplotlib nibabel nilearn pandas scipy
brew install awscli  # for OpenNeuro download
```

### Generate figures

```bash
cd figures
python3 generate_figures.py
```

### Compute D_eff on LSD dataset

```bash
python3 analysis/compute_dimensionality.py
```

### Compute spectral centroid

```bash
python3 analysis/compute_spectral_centroid.py
```

## Datasets

- **ds003059**: Carhart-Harris LSD dataset (N=15, placebo-controlled crossover)
- **ds006072**: Siegel psilocybin precision mapping (N=7, longitudinal)

Both available from OpenNeuro.

## Submission Status

![Status](https://img.shields.io/badge/Nature_Comms-Under_Review-yellow)
![Preprint](https://img.shields.io/badge/Preprint-In_Review_(Research_Square)-blue)

| Date | Event | Details |
|------|-------|---------|
| 2024-12-04 | Submitted | NCOMMS-25-98245 |
| 2024-12-04 | Initial screening | Editor not yet assigned |

**Manuscript:** NCOMMS-25-98245
**Type:** Article
**Corresponding Author:** Ian Todd (University of Sydney)
**Subject:** Computational Neuroscience / Cognitive Neuroscience

<!-- Update badge: Under_Review (yellow), Revision_Requested (orange), Accepted (green), Rejected (red) -->

## Citation

```bibtex
@misc{todd2025dimensionality,
  title={Psychedelics as Dimensionality Modulators: A Cortical Reservoir Theory of Serotonergic Plasticity},
  author={Todd, Ian},
  year={2025},
  note={Under review at Nature Communications (NCOMMS-25-98245)}
}
```

## Patent Notice

The **Brain Rate Variability (BRV)** metric and associated monitoring methods described herein are subject to **Australian Provisional Patent Application** (Ref: AMCZ-2515214626) filed by Coherence Dynamics Australia Pty Ltd. Commercial use requires licensing. Contact: ian@coherencedynamics.com

## License

MIT License
