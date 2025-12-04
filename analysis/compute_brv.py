#!/usr/bin/env python3
"""
Compute Brain Rate Variability (BRV) as metastability.

BRV is defined as the variance of the Kuramoto order parameter R(t),
which measures global synchronization across brain regions or channels.

High BRV indicates a system that continuously traverses between
synchronized and desynchronized states - the "dynamical flexibility"
characteristic of healthy cortical function and enhanced by psychedelics.

Reference:
    Todd, I. (2025). Psychedelics as Dimensionality Modulators.

See also:
    Deco, G., Kringelbach, M. L., et al. (2017). The dynamics of resting
    fluctuations in the brain: metastability and its dynamical cortical core.
"""

import numpy as np
from scipy.signal import hilbert, butter, filtfilt


def compute_brv_metastability(data, band=None, fs=None):
    """
    Compute Brain Rate Variability (BRV) as metastability.

    BRV is the variance of the Kuramoto order parameter R(t), which measures
    the instantaneous global synchronization level across channels.

    Parameters
    ----------
    data : ndarray, shape (n_timepoints, n_channels)
        Time series data (e.g., EEG channels or fMRI ROIs).
        For best results, data should be band-passed to the frequency
        range of interest (e.g., alpha 8-12 Hz for EEG, 0.01-0.1 Hz for fMRI).
    band : tuple of (low, high), optional
        If provided, band-pass filter the data to this frequency range (Hz).
        Requires fs to be specified.
    fs : float, optional
        Sampling frequency in Hz. Required if band is specified.

    Returns
    -------
    brv : float
        Brain Rate Variability (variance of Kuramoto order parameter).
        Higher values indicate greater metastability / dynamical flexibility.
    R_t : ndarray, shape (n_timepoints,)
        Time series of the Kuramoto order parameter R(t).
        R=1 means perfect synchronization, R=0 means complete desynchronization.

    Examples
    --------
    >>> # Synthetic example: synchronized vs desynchronized signals
    >>> import numpy as np
    >>> t = np.linspace(0, 10, 1000)
    >>>
    >>> # Low BRV: all channels synchronized
    >>> sync_data = np.column_stack([np.sin(2*np.pi*t + 0.1*i) for i in range(10)])
    >>> brv_sync, _ = compute_brv_metastability(sync_data)
    >>>
    >>> # High BRV: channels fluctuate between sync and desync
    >>> meta_data = np.column_stack([
    ...     np.sin(2*np.pi*t + np.sin(0.5*t)*i) for i in range(10)
    ... ])
    >>> brv_meta, _ = compute_brv_metastability(meta_data)
    >>>
    >>> print(f"Synchronized BRV: {brv_sync:.4f}")
    >>> print(f"Metastable BRV: {brv_meta:.4f}")
    """
    # Optionally band-pass filter
    if band is not None:
        if fs is None:
            raise ValueError("fs must be provided when band is specified")
        data = bandpass_filter(data, band[0], band[1], fs)

    # Get analytic signal via Hilbert transform to extract instantaneous phase
    analytic_signal = hilbert(data, axis=0)
    phases = np.angle(analytic_signal)  # Shape: (time, channels)

    # Compute Kuramoto order parameter R(t) at each timepoint
    # R(t) = |mean of complex phasors across channels|
    complex_phasors = np.exp(1j * phases)
    global_phasor = np.mean(complex_phasors, axis=1)  # Average over channels
    R_t = np.abs(global_phasor)  # Magnitude is synchronization level (0 to 1)

    # BRV is the variance of R(t)
    brv = np.var(R_t)

    return brv, R_t


def bandpass_filter(data, low, high, fs, order=4):
    """
    Apply zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    data : ndarray, shape (n_timepoints, n_channels)
        Data to filter.
    low : float
        Low cutoff frequency (Hz).
    high : float
        High cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    order : int
        Filter order.

    Returns
    -------
    filtered : ndarray
        Bandpass filtered data.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, data, axis=0)


def demo_brv():
    """
    Demonstrate BRV calculation with synthetic data showing
    different dynamical regimes.
    """
    print("="*60)
    print("Demo: Brain Rate Variability (BRV) as Metastability")
    print("="*60)

    np.random.seed(42)
    n_time = 2000
    n_channels = 50
    t = np.linspace(0, 20, n_time)

    # Regime 1: High synchronization (low BRV)
    # All channels oscillate together
    base_signal = np.sin(2 * np.pi * 1.0 * t)
    sync_data = np.column_stack([
        base_signal + 0.1 * np.random.randn(n_time)
        for _ in range(n_channels)
    ])

    # Regime 2: Complete desynchronization (low BRV)
    # All channels oscillate independently with random phases
    desync_data = np.column_stack([
        np.sin(2 * np.pi * 1.0 * t + np.random.uniform(0, 2*np.pi))
        + 0.1 * np.random.randn(n_time)
        for _ in range(n_channels)
    ])

    # Regime 3: Metastable (high BRV)
    # System fluctuates between synchronized and desynchronized states
    # This mimics what we expect under psychedelics
    phase_modulation = 3.0 * np.sin(0.2 * t)  # Slow modulation of phase spread
    meta_data = np.column_stack([
        np.sin(2 * np.pi * 1.0 * t + phase_modulation * (i/n_channels))
        + 0.1 * np.random.randn(n_time)
        for i in range(n_channels)
    ])

    # Compute BRV for each regime
    brv_sync, R_sync = compute_brv_metastability(sync_data)
    brv_desync, R_desync = compute_brv_metastability(desync_data)
    brv_meta, R_meta = compute_brv_metastability(meta_data)

    print("\nResults:")
    print(f"  Synchronized regime:    BRV = {brv_sync:.4f}, mean R = {R_sync.mean():.3f}")
    print(f"  Desynchronized regime:  BRV = {brv_desync:.4f}, mean R = {R_desync.mean():.3f}")
    print(f"  Metastable regime:      BRV = {brv_meta:.4f}, mean R = {R_meta.mean():.3f}")

    print("\nInterpretation:")
    print("  - Synchronized: R stays high (~1), variance low → low BRV")
    print("  - Desynchronized: R stays low (~0), variance low → low BRV")
    print("  - Metastable: R fluctuates widely → HIGH BRV")
    print("\n  Psychedelics are predicted to increase BRV by enabling")
    print("  transitions between synchronization states.")

    return {
        'sync': (brv_sync, R_sync),
        'desync': (brv_desync, R_desync),
        'meta': (brv_meta, R_meta)
    }


if __name__ == "__main__":
    demo_brv()
