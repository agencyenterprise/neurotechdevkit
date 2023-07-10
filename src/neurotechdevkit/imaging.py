"""Ultrasound imaging simulations."""

from typing import Optional, Tuple
import warnings

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def demodulate_rf_to_iq(rf_signals: np.ndarray,
                        freq_sampling: float,
                        *,
                        freq_carrier: Optional[float] = None,
                        bandwidth: Optional[float] = None,
                        time_offset: float = 0.0,
                        return_analytic: bool = False) -> Tuple[np.ndarray, float]:
    """
    Perform demodulation of radio-frequency (RF) signals to in-phase/quadrature (I/Q) signals.

    Parameters:
        rf_signals: Radio-frequency signals to be demodulated. Shape: (num_samples, num_channels)
        freq_sampling: Sampling frequency of the RF signals (in Hz).
        freq_carrier: Carrier frequency for down-mixing (in Hz). If not provided, it will be automatically calculated
            based on the RF signals.
        bandwidth: Bandwidth, expressed as a fraction (between 0 and 2) of the carrier frequency.
            If provided, it adjusts the cut-off frequency of the low-pass filter.
            If None, sets the cut-off frequency based on the carrier frequency.
        time_offset: Time offset (in seconds) to apply to the RF signals.
        return_analytic: Flag for returning the analytic signal. If True, the complex envelope of the demodulated
            signals will be returned. The complex envelope represents the baseband representation of the modulated
            signals, capturing the amplitude and phase information.

    Returns:
        - iq_signals: In-phase/quadrature (I/Q) components of the demodulated signals. Shape: (num_samples, num_channels)
        - freq_carrier: Carrier frequency used for down-mixing.

    Warnings:
        - If the RF signals are undersampled and harmful aliasing is suspected.

    Inspired by the MATLAB Ultrasound Toolbox (MUST) `rf2iq` function.
    """

    # Check input arguments
    assert np.isrealobj(rf_signals), "rf_signals must be real-valued."

    # Time vector
    assert rf_signals.ndim == 2, "Expected rf_signals to have shape (num_samples, num_channels)."
    num_samples, num_channels = rf_signals.shape
    time_arr = (np.arange(num_samples) / freq_sampling) + time_offset

    if freq_carrier is None:
        # Estimate the carrier frequency based on the power spectrum
        freq_carrier = _estimate_carrier_frequency(rf_signals, freq_sampling)

    # Normalized cut-off frequency
    if bandwidth is None:
        normalized_freq_cutoff = min(2 * freq_carrier / freq_sampling, 0.5)
    else:
        assert np.isscalar(bandwidth), "The signal bandwidth (bandwidth) must be a scalar."
        assert 0 < bandwidth < 2, "The signal bandwidth (bandwidth) must be within the interval of (0, 2)."
        normalized_freq_cutoff = freq_carrier * bandwidth / freq_sampling

    # Down-mix the RF signals
    # by multiplying them with a complex exponential corresponding to the carrier frequency.
    iq_signals = rf_signals * np.exp(-1j * 2 * np.pi * freq_carrier * time_arr[:, np.newaxis])

    # Low-pass Butterworth filter to extract demodulated I/Q signals
    sos = butter(5, normalized_freq_cutoff, output='sos')
    iq_signals = sosfiltfilt(sos, iq_signals, axis=0) * 2

    # Analytic signal: complex envelope
    if return_analytic:
        iq_signals = iq_signals * np.exp(1j * 2 * np.pi * freq_carrier * time_arr[:, np.newaxis])

    # Display a warning message if harmful aliasing is suspected
    if _potential_harmful_aliasing(freq_sampling, freq_carrier, bandwidth):
        warnings.warn("Harmful aliasing is present: the aliases are not mutually exclusive!", UserWarning)

    return iq_signals, freq_carrier


def _estimate_carrier_frequency(rf_signals: np.ndarray, freq_sampling: float, max_num_channels: int = 100) -> float:
    """Estimate the carrier frequency based on the power spectrum of the RF signals.

    Args:
        rf_signals: Radio-frequency signals estimate the carrier frequency of. Shape: (num_samples, num_channels)
        freq_sampling: Sampling frequency of the RF signals (in Hz).
        max_num_channels: Maximum number of channels to use for power spectrum analysis.
    """
    assert rf_signals.ndim == 2, "Expected rf_signals to have shape (num_samples, num_channels)."
    num_samples, num_channels = rf_signals.shape

    # randomly select channels (scan-lines) to speed up calculation
    selected_channel_idxs = np.random.choice(num_channels, min(max_num_channels, num_channels), replace=False)
    # Calculate the power spectrum using welch
    frequencies, power_spectrum = welch(rf_signals[:, selected_channel_idxs], fs=freq_sampling, axis=0)

    # Extract the positive portion of the power spectrum
    relevant_spectrum = power_spectrum[:num_channels // 2 + 1]
    relevant_frequencies = frequencies[:num_channels // 2 + 1]

    # Estimate the carrier frequency using the weighted average
    weighted_average = np.average(relevant_frequencies, weights=relevant_spectrum)
    # Convert from spectrum-index to Hz
    freq_carrier = weighted_average * freq_sampling / num_samples

    return freq_carrier


def _potential_harmful_aliasing(freq_sampling: float, freq_carrier: float, bandwidth: float):
    """Determine if harmful aliasing is present in the RF signals.

    Args:
        freq_sampling: Sampling frequency of the RF signals (in Hz).
        freq_carrier: Carrier frequency for down-mixing (in Hz).
        bandwidth: Bandwidth, expressed as a fraction (between 0 and 2) of the carrier frequency.

    Returns:
        True if harmful aliasing is suspected, False otherwise.
    """
    if freq_sampling < (2 * freq_carrier + bandwidth):  # The RF signal is undersampled
        freq_low = freq_carrier - bandwidth / 2
        freq_high = freq_carrier + bandwidth / 2
        n = np.floor(freq_high / (freq_high - freq_low)).astype(int)
        aliasing_freqs = 2 * freq_high / np.arange(1, n + 1)
        if np.any((aliasing_freqs <= freq_sampling) & (freq_sampling <= 2 * freq_low / np.arange(0, n))):
            return True

    return False
