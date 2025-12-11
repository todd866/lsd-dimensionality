# Psychedelics as Dimensionality Modulators

Code and analysis for the paper: **"Psychedelics as Dimensionality Modulators: A Cortical Reservoir Theory of Serotonergic Plasticity"**

Target journal: *Translational Psychiatry*

## Overview

This repository accompanies our framework proposing that classical psychedelics modulate the **effective dimensionality** of cortical dynamics through 5-HT2A-mediated dendritic gain amplification. MEG analysis reveals a mechanism-specific dissociation: psychedelics desynchronize, ketamine does not.

### Key Finding: Mechanism-Specific Dissociation

| Compound | N pairs | Coherence Change | p-value | Cohen's d |
|----------|---------|------------------|---------|-----------|
| Psilocybin | 20 | **-15.0%** | **0.003** | -0.78 |
| LSD | 15 | **-13.4%** | 0.082 | -0.50 |
| Ketamine | 18 | +5.7% | 0.290 | +0.26 |
| Tiagabine | 15 | +10.8% | 0.307 | +0.28 |

**Key insight**: Classical psychedelics (5-HT2A agonists) produce significant oscillatory desynchronization, while ketamine (NMDA antagonist) shows no effect. This specificity suggests that while both drug classes produce altered states, only serotonergic psychedelics function by dismantling the intrinsic oscillatory constraints of the cortex.

### fMRI Validation

| Dataset | Compound | D_eff Change | Spectral Centroid | p-value |
|---------|----------|--------------|-------------------|---------|
| ds003059 (N=15) | LSD | +8.6% | +10.0% | p=0.0008 |
| ds006072 (N=1) | Psilocybin | +19.2% | +18.6% | p=0.044 |

## Structure

```
├── lsd_dimensionality.tex               # Full manuscript (LaTeX)
├── lsd_dimensionality.pdf               # Compiled PDF
├── references.bib                       # Bibliography
├── cover_letter_translational_psychiatry.tex  # Cover letter
├── figures/                             # Publication figures
│   ├── generate_figures.py              # Figure generation script
│   ├── fig1_eigenmode_mechanism.pdf     # Conceptual framework
│   ├── fig2_three_phase_model.pdf       # Three-phase model
│   ├── fig3_meg_compound_comparison.pdf # MEG results (key figure)
│   └── fig4_mechanism_specificity.pdf   # Psilocybin vs ketamine
├── analysis/                            # Analysis code
│   ├── compute_dimensionality.py        # D_eff for LSD fMRI
│   ├── compute_meg_deff.py              # D_eff for MEG data
│   ├── compute_spectral_centroid.py     # Spectral centroid
│   └── download_lsd_data.py             # Data download helper
└── data/                                # Data (not tracked)
```

## Key Concepts

- **Effective Dimensionality (D_eff)**: Participation ratio of covariance eigenvalues
- **Oscillatory Coherence**: MEG-derived measure of synchronized cortical activity
- **Three-Phase Model**: Overshoot → Refractory → Recanalization
- **Mechanism**: 5-HT2A activation → dendritic gain amplification → desynchronization

## Running the Analysis

### Install dependencies

```bash
pip install numpy matplotlib nibabel nilearn pandas scipy
```

### Generate figures

```bash
cd figures
python3 generate_figures.py
```

### Compute MEG D_eff

```bash
python3 analysis/compute_meg_deff.py
```

## Datasets

- **MEG**: Muthukumaraswamy et al. (2013) - LSD, psilocybin, ketamine, tiagabine
- **ds003059**: Carhart-Harris LSD fMRI (N=15, placebo-controlled crossover)
- **ds006072**: Siegel psilocybin precision mapping (N=7, longitudinal)

All available from OpenNeuro or original authors.

## Submission Status

![Status](https://img.shields.io/badge/Translational_Psychiatry-Submitted-yellow)

| Date | Event | Details |
|------|-------|---------|
| 2025-12-04 | Submitted | Nature Communications (NCOMMS-25-98245) |
| 2025-12-10 | Desk rejection | NC - "better fit for specialist journal" |
| 2025-12-11 | Resubmitted | Translational Psychiatry |

**Corresponding Author:** Ian Todd (University of Sydney)

## Interactive Simulation

Try the companion simulation at [coherencedynamics.com/simulations/lsd-landscape](https://coherencedynamics.com/simulations/lsd-landscape) — drag to modulate 5-HT2A gain and watch cortical oscillators desynchronize.

## Citation

```bibtex
@misc{todd2025dimensionality,
  title={Psychedelics as Dimensionality Modulators: A Cortical Reservoir Theory of Serotonergic Plasticity},
  author={Todd, Ian},
  year={2025},
  note={Under review at Translational Psychiatry}
}
```

## Patent Notice

The **Brain Rate Variability (BRV)** metric and associated monitoring methods described herein are subject to **Australian Provisional Patent Application** (Ref: AMCZ-2515214626) filed by Coherence Dynamics Australia Pty Ltd. Commercial use requires licensing. Contact: ian@coherencedynamics.com

## License

MIT License
