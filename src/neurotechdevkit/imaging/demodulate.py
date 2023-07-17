"""Demodulate ultrasound radio-frequency (RF) signals."""

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch


def demodulate_rf_to_iq(
    rf_signals: np.ndarray,
    freq_sampling: float,
    *,
    freq_carrier: Optional[float] = None,
    bandwidth: Optional[float] = None,
    time_offset: float = 0.0,
    return_analytic: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Demodulate radio-frequency (RF) signals to in-phase/quadrature (I/Q) signals.

    Performed as one-step of ultrasound imaging.

    Parameters:
        rf_signals: Radio-frequency signals to be demodulated.
            Shape: (num_samples, num_channels)
        freq_sampling: Sampling frequency of the RF signals (in Hz).
        freq_carrier: Carrier frequency for down-mixing (in Hz). If not provided,
            it will be automatically calculated based on the RF signals.
        bandwidth: Bandwidth, expressed as a fraction (between 0 and 2) of the
            carrier frequency.
            If provided, it adjusts the cut-off frequency of the low-pass filter.
            If None, sets the cut-off frequency based on the carrier frequency.
        time_offset: Time offset (in seconds) to apply to the RF signals.
        return_analytic: Flag for returning the analytic signal.
            If True, the complex envelope of the demodulated signals will be
            returned. The complex envelope represents the baseband representation
            of the modulated signals, capturing the amplitude and phase information.

    Returns:
        - iq_signals: In-phase/quadrature (I/Q) components of the demodulated signals.
            Shape: (num_samples, num_channels)
        - freq_carrier: Carrier frequency used for down-mixing.

    Warnings:
        - If the RF signals are undersampled and harmful aliasing is suspected.

    Inspired by the MATLAB Ultrasound Toolbox (MUST) `rf2iq` function.
    """

    # Check input arguments
    assert np.isrealobj(rf_signals), "rf_signals must be real-valued."

    # Time vector
    assert (
        rf_signals.ndim == 2
    ), "Expected rf_signals to have shape (num_samples, num_channels)."
    num_samples, num_channels = rf_signals.shape
    time_arr = (np.arange(num_samples) / freq_sampling) + time_offset

    if freq_carrier is None:
        # Estimate the carrier frequency based on the power spectrum
        freq_carrier = _estimate_carrier_frequency(rf_signals, freq_sampling)

    # Normalized cut-off frequency
    if bandwidth is None:
        normalized_freq_cutoff = 2 * freq_carrier / freq_sampling
        # Cutoff should be less than the (normalized) Nyquist frequency
        normalized_freq_cutoff = min(normalized_freq_cutoff, 0.5)
        bandwidth = normalized_freq_cutoff * freq_sampling / freq_carrier
    else:
        assert np.isscalar(
            bandwidth
        ), "The signal bandwidth (bandwidth) must be a scalar."
        assert (
            0 < bandwidth < 2
        ), "The signal bandwidth (bandwidth) must be within the interval of (0, 2)."
        normalized_freq_cutoff = freq_carrier * bandwidth / freq_sampling

    # Down-mix the RF signals
    # by multiplying them with a complex exponential corresponding to the
    # carrier frequency.
    iq_signals = rf_signals * np.exp(
        -1j * 2 * np.pi * freq_carrier * time_arr[:, np.newaxis]
    )

    # Low-pass Butterworth filter to extract demodulated I/Q signals
    sos = butter(5, normalized_freq_cutoff, output="sos")
    iq_signals = sosfiltfilt(sos, iq_signals, axis=0) * 2

    # Analytic signal: complex envelope
    if return_analytic:
        iq_signals = iq_signals * np.exp(
            1j * 2 * np.pi * freq_carrier * time_arr[:, np.newaxis]
        )

    # Display a warning message if harmful aliasing is suspected
    if _potential_harmful_aliasing(freq_sampling, freq_carrier, bandwidth):
        warnings.warn(
            "Harmful aliasing is present: the aliases are not mutually exclusive!",
            UserWarning,
        )

    return iq_signals, freq_carrier


def _estimate_carrier_frequency(
    rf_signals: np.ndarray,
    freq_sampling: float,
    max_num_channels: int = 100,
    use_welch: bool = True,
) -> float:
    """Estimate the carrier frequency based on the power spectrum of the RF signals.

    Args:
        rf_signals: Radio-frequency signals estimate the carrier frequency of.
            Shape: (num_samples, num_channels)
        freq_sampling: Sampling frequency of the RF signals (in Hz).
        max_num_channels: Maximum number of channels to use for power spectrum analysis.
        use_welch: whether to use Welch's method to estimate the power spectrum.
            If False, uses FFT sum-of-squares.
    """
    assert (
        rf_signals.ndim == 2
    ), "Expected rf_signals to have shape (num_samples, num_channels)."
    assert np.isrealobj(rf_signals), "rf_signals must be real-valued."
    num_samples, num_channels = rf_signals.shape

    # randomly select channels (scan-lines) to speed up calculation
    num_selected_channels = min(max_num_channels, num_channels)
    selected_channel_idxs = np.random.choice(
        num_channels, num_selected_channels, replace=False
    )

    # Calculate the power spectrum
    kwargs = {} if use_welch else {"nperseg": num_samples, "window": "boxcar"}
    frequencies, power_spectrum = welch(
        rf_signals[:, selected_channel_idxs],
        fs=freq_sampling,
        scaling="spectrum",
        return_onesided=True,  # only positive frequencies
        axis=0,
        **kwargs,
    )

    # Aggregate spectrum across channels
    (num_frequencies,) = frequencies.shape
    assert power_spectrum.shape == (num_frequencies, num_selected_channels)
    power_spectrum = power_spectrum.sum(axis=1)
    # Estimate the carrier frequency using the weighted average
    freq_carrier = np.average(frequencies, weights=power_spectrum)

    # Calculated carrier frequency should be positive
    assert freq_carrier > 0, "Estimated carrier frequency is negative: {}".format(
        freq_carrier
    )

    return freq_carrier


def _potential_harmful_aliasing(
    freq_sampling: float, freq_carrier: float, bandwidth: float
):
    """Determine if harmful aliasing is present in the RF signals.

    See: https://en.wikipedia.org/wiki/Undersampling

    Args:
        freq_sampling: Sampling frequency of the RF signals (in Hz).
        freq_carrier: Carrier frequency for down-mixing (in Hz).
        bandwidth: Bandwidth, expressed as a fraction (between 0 and 2) of the
            carrier frequency.

    Returns:
        True if harmful aliasing is suspected, False otherwise.
    """
    if freq_sampling < (2 * freq_carrier + bandwidth):  # The RF signal is undersampled
        freq_low = freq_carrier - bandwidth / 2
        freq_high = freq_carrier + bandwidth / 2
        n = np.floor(freq_high / (freq_high - freq_low))
        aliasing_freqs = 2 * freq_high / np.arange(1, n + 1)
        avoids_overlap = np.any(
            ((2 * freq_high / np.arange(1, n + 1)) <= freq_sampling)
            & (freq_sampling <= 2 * freq_low / np.arange(n))
        )
        return not avoids_overlap

    return False
