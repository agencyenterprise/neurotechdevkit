"""Beam-form I/Q signals for ultrasound imaging."""

from enum import IntEnum
from typing import Optional

import numpy as np
import xarray as xr
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_array


class InterpolationMethod(IntEnum):
    """Interpolation methods for beamforming.

    Value corresponds to the number of points used for interpolation.
    """

    NEAREST = 1  # nearest-neighbor interpolation
    LINEAR = 2
    QUADRATIC = 3
    LANZCOS_3 = 3  # 3-lobe Lanczos interpolation
    LSQ_PARABOLA_5 = 5  # 5-point least-squares parabolic interpolation
    LANZCOS_5 = 6  # 5-lobe Lanczos interpolation


def beamform_delay_and_sum(
    iq_signals: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    fs: float,
    fc: float,
    **kwargs,
) -> np.ndarray:
    """Delay-And-Sum beamforming.

    Currently, assumes that the input signal is in-phase/quadrature (I/Q) signals.

    Parameters:
        iq_signals: 2-D complex-valued array containing I/Q signals.
            Shape: (time_samples, num_channels)
            channels usually correspond to transducer elements.
            TODO: Support 3rd dimension for repeated-echos (slow-time)
        x: 2-D array specifying the x-coordinates of the [x, z] image grid.
        z: 2-D array specifying the z-coordinates of the [x, z] image grid.
        fs: sampling frequency of the input signals.
        fc: center/carrier frequency of the input signals.
        **kwargs: additional arguments passed to `delay_and_sum_matrix`

    Returns:
        Beamformed signals at the specified [x, z] image grid.

    Notes:
        - For a linear array: x = 0, z = 0 corresponds to the center of the transducer.
            The x-axis is parallel to the transducer, and the z-axis is perpendicular
            to the transducer, with z increasing in the direction of the transducer.
        - Memory usage for the delay-and-sum matrix is:
            O(num_pixels * num_channels * num_interpolation_points).

    Adapted from: https://doi.org/10.1016/j.ultras.2020.106309
    """
    # Check input arguments
    assert (
        iq_signals.ndim == 2
    ), "Expected iq_signals to have shape (time_samples, num_channels)."
    assert np.iscomplexobj(
        iq_signals
    ), "Expected iq_signals to be complex-valued I/Q signals."
    assert x.ndim == 2, "Expected x to have shape (width_pixels, depth_pixels)."
    assert z.ndim == 2, "Expected z to have shape (width_pixels, depth_pixels)."
    assert x.shape == z.shape, "Expected image grid x and z to have the same shape."
    num_time_samples, num_channels = iq_signals.shape

    # Check for potential underlying errors before allocating memory
    das_matrix: csr_array = delay_and_sum_matrix(
        num_time_samples=num_time_samples,
        num_channels=num_channels,
        x=x,
        z=z,
        fs=fs,
        fc=fc,
        **kwargs,
    )

    # Beamform
    iq_signals_column_vec = iq_signals.reshape(-1, 1)
    beamformed_iq_signals = das_matrix @ iq_signals_column_vec
    beamformed_iq_signals = beamformed_iq_signals.reshape(x.shape)

    return beamformed_iq_signals


def delay_and_sum_matrix(
    num_time_samples: int,
    num_channels: int,
    x: np.ndarray,
    z: np.ndarray,
    *,
    pitch: float,  # m  center-to-center distance between two adjacent elements/channels
    tx_delays: np.ndarray,
    fs: float,  # Hz
    fc: float,  # Hz
    t0: float = 0,  # s
    c: float = 1540,  # m/s
    f_number: Optional[float] = 1.5,
    width: Optional[float] = None,  # m
    bandwidth: Optional[float] = 0.75,
    method: InterpolationMethod = InterpolationMethod.NEAREST,
) -> csr_array:
    """
    Calculates the delay-and-sum sparse matrix for beamforming I/Q signals.

    Returns:
        delay-and-sum matrix. Shape: (num_pixels, num_samples*num_channels)

    Assumptions:
        - The input signal is in-phase/quadrature (I/Q) signals.
        - The input signal is sampled at a constant rate.
        - The sensor array is linear.
        - The sensor array is parallel to the x-axis and centered at (0, 0).
        - Nearest-neighbor interpolation is used for the delay-and-sum matrix.

    Notes:
        - Memory usage for the delay-and-sum matrix is:
            O(num_pixels * num_channels * num_interpolation_points).

    For details on delay-and-sum by matrix-multiplication:
        https://doi.org/10.1016/j.ultras.2020.106309
    """
    # Check inputs
    assert num_time_samples > 0, "Number of time samples must be positive."
    assert num_channels > 0, "Number of channels/elements must be positive."
    assert (x.size > 0) and (z.size > 0), "Number of pixels must be positive."
    assert x.ndim == 2, "Expected x to have shape (width_pixels, depth_pixels)."
    assert z.ndim == 2, "Expected z to have shape (width_pixels, depth_pixels)."
    assert x.shape == z.shape, "Expected image grid x and z to have the same shape."
    assert len(tx_delays) == num_channels, "Expected one transmit-delay per channel."
    if f_number is None:
        # Optimize f-number based on other parameters
        assert width is not None, "Element width is required to estimate f-number."
        assert (
            bandwidth is not None
        ), "Element bandwidth is required to estimate f-number."
        assert 0 < bandwidth < 2, "Fractional bandwidth at -6dB must be in (0, 2)."

        # Eq. 14 of https://doi.org/10.1016/j.ultras.2020.106309
        wavelength_min = c / (fc * (1 + bandwidth / 2))

        # Eq. 11 of https://doi.org/10.1016/j.ultras.2020.106309
        def directivity(theta, width, wavelength):
            return np.cos(theta) * np.sinc(width / wavelength * np.sin(theta))
        optimal_directivity = 0.71

        # Eq. 13 of https://doi.org/10.1016/j.ultras.2020.106309
        result = minimize_scalar(
            lambda theta: np.abs(directivity(theta, width, wavelength_min) - optimal_directivity),
            bounds=(0, np.pi / 2),
        )
        alpha = result.x
        f_number = 1 / (2 * np.tan(alpha))
    else:
        assert f_number >= 0, "f-number must be non-negative."
    if method != InterpolationMethod.NEAREST:
        raise NotImplementedError("Only nearest-neighbor interpolation is supported.")

    # Helper parameters
    width_pixels, depth_pixels = x.shape
    num_pixels = x.size

    # Convert to xarray for more intuitive broadcasting
    x = xr.DataArray(x, dims=("x", "z"))  # keep as 0-index
    z = xr.DataArray(z, dims=("x", "z"))
    tx_delays = xr.DataArray(tx_delays, dims=("channel",))

    # transducer centered on (0, 0) parallel to x-axis
    x_channels = (np.arange(num_channels) - (num_channels - 1) / 2) * pitch
    x_channels = xr.DataArray(x_channels, dims=("channel",))  # keep as 0-index
    z_channels = np.zeros(x_channels.shape)
    z_channels = xr.DataArray(z_channels, dims=("channel",))

    # Effective transmit-distance from each (interpolated) element to each pixel
    # TODO: interpolate transducers for better estimation of delays
    # For now, don't interpolate
    x_channels_interp = x_channels
    z_channels_interp = z_channels
    tx_distance = tx_delays * c + np.sqrt(
        (x - x_channels_interp) ** 2 + (z - z_channels_interp) ** 2
    )
    assert tx_distance.sizes == {
        "x": width_pixels,
        "z": depth_pixels,
        "channel": num_channels,
    }, "Expected one transmit distance per pixel and channel."
    # transmit-pulse reaches pixel, based on the "closest" transducer channel/element
    tx_distance = tx_distance.min(dim="channel")
    assert tx_distance.sizes == {
        "x": width_pixels,
        "z": depth_pixels,
    }, "Expected transmit delays to be calculated for each pixel."

    # Receive-delay:
    # Calculate distances from each pixel to each receive-element/channel
    rx_distance = np.sqrt((x - x_channels) ** 2 + (z - z_channels) ** 2)
    assert rx_distance.sizes == {
        "x": width_pixels,
        "z": depth_pixels,
        "channel": num_channels,
    }, "Expected one receive delay per pixel and channel."

    # Wave propagation times
    tau = (tx_distance + rx_distance) / c

    # Convert wave propagation time to sample index (fast-time)
    # Keep `tau` for later use
    das_ds = tau.to_dataset(name="tau")
    das_ds["time_idx"] = (tau - t0) * fs
    assert das_ds.sizes == {
        "x": width_pixels,
        "z": depth_pixels,
        "channel": num_channels,
    }, "Expected one fast-time index and weight per pixel/channel combo."

    # Each channel only needs to consider pixels within its receive aperture
    # (i.e., angle-of-view = 2*alpha)
    if f_number > 0:
        # assume linear array at z=0
        half_aperture = z / (2 * f_number)
        is_within_aperture = np.abs(x - x_channels) <= half_aperture
        assert is_within_aperture.sizes == {
            "x": width_pixels,
            "z": depth_pixels,
            "channel": num_channels,
        }, "Expected to know whether each pixel was within each channel's aperture."
        das_ds = das_ds.where(is_within_aperture)
    else:
        assert f_number == 0, "f-number must be non-negative."

    # Use nearest-neighbor interpolation
    assert method == InterpolationMethod.NEAREST
    interp_weights = xr.DataArray([1], dims=("interp",))
    das_ds["time_idx"] = das_ds["time_idx"].round()
    is_valid_time_idx = (das_ds["time_idx"] >= 0) & (
        das_ds["time_idx"] <= (num_time_samples - 1)
    )
    das_ds = das_ds.where(is_valid_time_idx)

    if das_ds["time_idx"].count() == 0:
        raise ValueError(
            "No I/Q time indices correspond to valid measurements within the "
            "image grid."
        )

    # Rotate phase of I/Q signals, based on delays
    das_ds["weights"] = interp_weights * np.exp(1j * (2 * np.pi * fc) * das_ds.tau)
    # For nearest-neighbor,
    # we can drop interp dimension simply because we only have one interp point
    das_ds = das_ds.squeeze("interp")

    # In case, e.g., "x", "z", and "channel" are not 0-indexed
    # Convert "x", "z", "channel" coords to 0-indices, in case they have been
    # set to something else
    das_ds = das_ds.drop_indexes(das_ds.indexes.keys()).reset_coords(drop=True)

    # Convert from semi-sparse (x, z, channel)-shape array of time indices
    # to list of [x, z, time, channel] indices, and corresponding weights
    tmp_dim_order = ("x", "z", "channel")
    das_ds_flat = das_ds.stack(
        nonzero=tmp_dim_order,
    ).dropna(dim="nonzero", how="all")
    assert (
        das_ds_flat["time_idx"].notnull().all()
    ), "Should have dropped nuull time indices"
    assert das_ds_flat["weights"].notnull().all(), "Should have dropped null weights."
    assert len(das_ds_flat.dims) == 1, "Expected to have stacked all dimensions"
    assert das_ds_flat.count().equals(das_ds.count()), ".dropna shouldn't change count."

    x_z_multi_indices = np.row_stack((das_ds_flat.x, das_ds_flat.z))
    np.testing.assert_allclose(
        das_ds_flat.time_idx,
        das_ds_flat.time_idx.astype(int),
        rtol=0,
        atol=1e-6,
        err_msg="Expected time indices to be integers.",
    )
    time_channel_multi_indices = np.row_stack(
        (das_ds_flat.time_idx.astype(int), das_ds_flat.channel)
    )

    # Convert from [x, z, time, channel] indices to [x_z, time_channel] indices
    x_z_flat_indices = np.ravel_multi_index(
        x_z_multi_indices, (width_pixels, depth_pixels)
    )
    time_channel_flat_indices = np.ravel_multi_index(
        time_channel_multi_indices, (num_time_samples, num_channels)
    )

    # Construct delay-and-sum sparse matrix
    shape = (num_pixels, num_time_samples * num_channels)
    das_matrix = csr_array(
        (das_ds_flat["weights"].values, (x_z_flat_indices, time_channel_flat_indices)),
        shape=shape,
    )

    return das_matrix
