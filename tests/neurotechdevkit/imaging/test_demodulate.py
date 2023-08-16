import pytest
import numpy as np

from neurotechdevkit.imaging.demodulate import (
    demodulate_rf_to_iq,
    _estimate_carrier_frequency,
    _potential_harmful_aliasing,
)


def test_demodulate_rf_to_iq():
    rf_signals = np.random.rand(100, 2)
    freq_sampling = 1.0

    # Test with default parameters
    iq_signals, freq_carrier = demodulate_rf_to_iq(rf_signals, freq_sampling)
    assert iq_signals.shape == rf_signals.shape
    assert freq_carrier > 0
    assert np.iscomplexobj(iq_signals)

    # Test with specified freq_carrier and bandwidth
    iq_signals, freq_carrier = demodulate_rf_to_iq(rf_signals, freq_sampling, freq_carrier=0.5, bandwidth=0.1)
    assert iq_signals.shape == rf_signals.shape
    assert freq_carrier == 0.5
    assert np.iscomplexobj(iq_signals)


def test_estimate_carrier_frequency():
    rf_signals = np.random.rand(100, 2)
    freq_sampling = 1.0

    # Test with default parameters
    freq_carrier = _estimate_carrier_frequency(rf_signals, freq_sampling)
    assert freq_carrier > 0

    # Test with specified max_num_channels and use_welch
    freq_carrier = _estimate_carrier_frequency(rf_signals, freq_sampling, max_num_channels=1, use_welch=False)
    assert freq_carrier > 0

    # Test that freq_carrier matches when signal is sinusoidal
    time = np.arange(100) / freq_sampling
    freq_carrier = 0.1
    rf_signals = np.sin(2 * np.pi * freq_carrier * time)[:, np.newaxis]
    assert pytest.approx(freq_carrier, rel=0.01) == _estimate_carrier_frequency(rf_signals, freq_sampling)


def test_potential_harmful_aliasing():
    freq_carrier = 0.5
    freq_sampling = 0.5
    bandwidth = 0.1

    # Test with aliasing
    assert _potential_harmful_aliasing(freq_sampling, freq_carrier, bandwidth)

    # Test without aliasing
    freq_sampling = 1.5
    assert not _potential_harmful_aliasing(freq_sampling, freq_carrier, bandwidth)
