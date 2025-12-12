# Timescale-Dependent Cortical Dimensionality

Code and analysis for: **"Timescale-Dependent Cortical Dimensionality: Psychedelics Desynchronize Fast Oscillations but Spare the Slow Hemodynamic Manifold"**

Target: *Network Neuroscience* or *BioSystems*

## Key Finding: Timescale Dissociation

Psychedelics expand dimensionality at fast timescales (MEG) but not slow timescales (fMRI).

### MEG Results (136 sessions)

| Compound | Coherence Change | p-value | Cohen's d |
|----------|------------------|---------|-----------|
| Psilocybin | **-15.0%** | **0.003** | -0.78 |
| LSD | **-13.4%** | 0.082 | -0.50 |
| Ketamine | +5.7% | 0.290 | +0.26 |
| Tiagabine | +10.8% | 0.307 | +0.28 |

Classical psychedelics (5-HT2A agonists) produce significant desynchronization; ketamine (NMDA antagonist) does not.

### fMRI Results (N=7, 124 sessions)

| Metric | Baseline | Drug | Change | p-value | Cohen's d |
|--------|----------|------|--------|---------|-----------|
| D_eff | 51.5 ± 7.4 | 49.0 ± 12.4 | -5.7% | 0.47 | -0.32 |
| Spectral Centroid | 80.8 ± 9.5 | 93.7 ± 16.3 | **+16.8%** | 0.09 | **+0.82** |

D_eff unchanged, but spectral centroid shifts toward higher modes—variance redistributes even if dimensionality doesn't.

## Structure

```
├── lsd_dimensionality.tex      # Manuscript
├── lsd_dimensionality.pdf      # Compiled PDF
├── references.bib              # Bibliography
├── analysis/                   # Analysis code
│   ├── compute_meg_deff.py
│   ├── compute_psilocybin_all_subjects.py
│   └── compute_spectral_centroid.py
├── data/                       # Results (raw data not tracked)
│   ├── meg_deff_results.csv
│   └── psilocybin_all_subjects_results.csv
└── figures/
```

## Methods

- **D_eff** (participation ratio): `(Σλ)² / Σλ²`
- **Spectral centroid**: `Σ(i·λᵢ) / Σλᵢ`

## Datasets

- **MEG**: Muthukumaraswamy et al. (2013) - LSD, psilocybin, ketamine, tiagabine
- **fMRI**: Siegel psilocybin precision mapping (OpenNeuro ds006072)

## Interactive Simulation

[coherencedynamics.com/simulations/lsd-landscape](https://coherencedynamics.com/simulations/lsd-landscape)

## License

MIT
