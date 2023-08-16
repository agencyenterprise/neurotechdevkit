"""Helper functions for imaging module."""

import numpy as np


def log_compress(iq_signals_beamformed, dynamic_range_db: float = 40):
    """Log-compress the beamformed IQ signals."""
    assert np.iscomplexobj(iq_signals_beamformed), "I/Q signals should be complex"

    real_envelope = np.abs(iq_signals_beamformed)
    real_envelope[real_envelope == 0] = np.finfo(
        float
    ).eps  # add tiny offset to allow log10
    image_db = 20 * np.log10(real_envelope / np.max(real_envelope)) + dynamic_range_db
    image_db = np.clip(image_db / dynamic_range_db, 0, None)

    assert ((0 <= image_db) & (image_db <= 1)).all(), "Expected values in range [0, 1]"
    # Range [0, 1] corresponds to [-dynamic_range_db dB, 0 dB]

    return image_db


def gamma_compress(iq_signals_beamformed, exponent: float = 0.5):
    """Gamma-compress the beamformed IQ signals."""
    assert np.iscomplexobj(iq_signals_beamformed), "I/Q signals should be complex"

    real_envelope = np.abs(iq_signals_beamformed)
    normalized = real_envelope / real_envelope.max()
    image_gamma = np.power(normalized, exponent)

    return image_gamma
