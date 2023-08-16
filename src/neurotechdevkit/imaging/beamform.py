"""Beam-form I/Q signals for ultrasound imaging."""

from enum import IntEnum, unique
from typing import Optional

import numpy as np
import numpy.typing as npt
import xarray as xr
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_array


@unique
class InterpolationMethod(IntEnum):
    """Interpolation methods for beamforming.

    Value corresponds to the number of points used for interpolation.
    """

    NEAREST = 1  # nearest-neighbor interpolation
    LINEAR = 2
    QUADRATIC = 3
    LANZCOS_3 = 4  # 3-lobe Lanczos interpolation
    LSQ_PARABOLA_5 = 5  # 5-point least-squares parabolic interpolation
    LANZCOS_5 = 6  # 5-lobe Lanczos interpolation


def beamform_delay_and_sum(
    iq_signals: npt.NDArray[np.float_],
    x: npt.NDArray[np.float_],
    z: npt.NDArray[np.float_],
    freq_sampling: float,
    freq_carrier: float,
    **kwargs,
) -> npt.NDArray[np.float_]:
    """Delay-And-Sum beamforming.

    Currently, assumes that the input signal is in-phase/quadrature (I/Q) signals.

    Parameters:
        iq_signals: 2-D complex-valued array containing I/Q signals.
            Shape: (time_samples, num_channels) or
                (time_samples, num_channels, num_echoes)
            channels usually correspond to transducer elements.
        x: 2-D array specifying the x-coordinates of the [x, z] image grid.
        z: 2-D array specifying the z-coordinates of the [x, z] image grid.
        freq_sampling: sampling frequency of the input signals.
        freq_carrier: center/carrier frequency of the input signals.
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
    if iq_signals.ndim == 2:
        num_time_samples, num_channels = iq_signals.shape
    elif iq_signals.ndim == 3:
        num_time_samples, num_channels, num_echoes = iq_signals.shape
    else:
        raise ValueError(
            "Expected iq_signals to have shape (time_samples, num_channels) or "
            "(time_samples, num_channels, num_echoes)."
        )

    assert np.iscomplexobj(
        iq_signals
    ), "Expected iq_signals to be complex-valued I/Q signals."
    assert x.ndim == 2, "Expected x to have shape (width_pixels, depth_pixels)."
    assert z.ndim == 2, "Expected z to have shape (width_pixels, depth_pixels)."
    assert x.shape == z.shape, "Expected image grid x and z to have the same shape."

    # Check for potential underlying errors before allocating memory
    das_matrix: csr_array = delay_and_sum_matrix(
        num_time_samples=num_time_samples,
        num_channels=num_channels,
        x=x,
        z=z,
        freq_sampling=freq_sampling,
        freq_carrier=freq_carrier,
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
    x: npt.NDArray[np.float_],
    z: npt.NDArray[np.float_],
    *,
    pitch: float,  # m  center-to-center distance between two adjacent elements/channels
    tx_delays: npt.NDArray[np.float_],
    freq_sampling: float,  # Hz
    freq_carrier: float,  # Hz
    start_time: float = 0,  # s
    speed_sound: float = 1540,  # m/s
    f_number: Optional[float] = 1.5,
    width: Optional[float] = None,  # m
    bandwidth: Optional[float] = 0.75,
    method: InterpolationMethod = InterpolationMethod.LINEAR,
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

    Notes:
        - Alternatively, delay-and-sum can be implemented using an explicit
            for-loop over the delays. However, Python for-loops are much
            slower than scipy's optimized sparse matrix multiplication.
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
        wavelength_min = speed_sound / (freq_carrier * (1 + bandwidth / 2))

        # Eq. 11 of https://doi.org/10.1016/j.ultras.2020.106309
        def directivity(theta, width, wavelength):
            return np.cos(theta) * np.sinc(width / wavelength * np.sin(theta))

        optimal_directivity = 0.71

        # Eq. 13 of https://doi.org/10.1016/j.ultras.2020.106309
        result = minimize_scalar(
            lambda theta: np.abs(
                directivity(theta, width, wavelength_min) - optimal_directivity
            ),
            bounds=(0, np.pi / 2),
        )
        alpha = result.x
        f_number = 1 / (2 * np.tan(alpha))
    else:
        assert f_number >= 0, "f-number must be non-negative."

    # Helper parameters
    width_pixels, depth_pixels = x.shape
    num_pixels = x.size

    # Convert to xarray for more intuitive broadcasting
    x_dataarray = xr.DataArray(x, dims=("x", "z"))  # keep as 0-index
    z_dataarray = xr.DataArray(z, dims=("x", "z"))
    tx_delays_dataarray = xr.DataArray(tx_delays, dims=("channel",))

    # transducer centered on (0, 0) parallel to x-axis
    x_channels = xr.DataArray(
        (np.arange(num_channels) - (num_channels - 1) / 2) * pitch,
        dims=("channel",)
    )  # keep as 0-index
    z_channels = xr.DataArray(
        np.zeros(x_channels.shape),
        dims=("channel",)
    )

    # Effective transmit-distance from each (interpolated) element to each pixel
    # TODO: interpolate transducers for better estimation of delays
    # For now, don't interpolate
    x_channels_interp = x_channels
    z_channels_interp = z_channels
    tx_distance = tx_delays_dataarray * speed_sound + np.sqrt(
        (x_dataarray - x_channels_interp) ** 2 + (z_dataarray - z_channels_interp) ** 2
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
    rx_distance = np.sqrt((x_dataarray - x_channels) ** 2 + (z_dataarray - z_channels) ** 2)
    assert rx_distance.sizes == {  # type: ignore  # np.sqrt return-typed as numpy despite xarray patching: https://github.com/pydata/xarray/issues/6524
        "x": width_pixels,
        "z": depth_pixels,
        "channel": num_channels,
    }, "Expected one receive delay per pixel and channel."

    # Wave propagation times
    tau = (tx_distance + rx_distance) / speed_sound

    # Convert wave propagation time to sample index (fast-time)
    # Keep `tau` for later use
    das_ds = tau.to_dataset(name="tau")
    das_ds["time_idx"] = (tau - start_time) * freq_sampling
    assert das_ds.sizes == {
        "x": width_pixels,
        "z": depth_pixels,
        "channel": num_channels,
    }, "Expected one fast-time index and weight per pixel/channel combo."

    # Each channel only needs to consider pixels within its receive aperture
    # (i.e., angle-of-view = 2*alpha)
    if f_number > 0:
        # assume linear array at z=0
        half_aperture = z_dataarray / (2 * f_number)
        is_within_aperture = np.abs(x_dataarray - x_channels) <= half_aperture
        assert is_within_aperture.sizes == {  # type: ignore  # np.abs return-typed as numpy despite xarray patching: https://github.com/pydata/xarray/issues/6524
            "x": width_pixels,
            "z": depth_pixels,
            "channel": num_channels,
        }, "Expected to know whether each pixel was within each channel's aperture."
        das_ds = das_ds.where(is_within_aperture)
    else:
        assert f_number == 0, "f-number must be non-negative."

    # We have discrete time samples, which do not correspond exactly to the
    # pixel grid. Interpolate to find the pixel grid values
    if method == InterpolationMethod.NEAREST:
        interp_weights = xr.DataArray([1], dims=("interp",))
        is_valid_time_idx = (das_ds["time_idx"] >= 0) & (
            das_ds["time_idx"] <= (num_time_samples - 1)
        )
        das_ds = das_ds.where(is_valid_time_idx)
        das_ds["time_idx_round"] = das_ds["time_idx"].round()
    elif method == InterpolationMethod.LINEAR:
        # E.g., a time index of 2.3 gets 0.7*pixel_2 + 0.3*pixel_3
        is_valid_time_idx = (das_ds["time_idx"] >= 0) & (
            das_ds["time_idx"] <= (num_time_samples - 1)
        )
        das_ds = das_ds.where(is_valid_time_idx)
        # Weighted sum of two nearest time samples
        interp_weights = xr.concat(
            [
                1 - (das_ds["time_idx"] % 1),  # how close to upper time sample?
                das_ds["time_idx"] % 1,  # how close to lower time sample?
            ],
            dim="interp",
        )
        das_ds["time_idx_round"] = xr.concat(  # type: ignore  # np.floor and np.ceil return-typed as numpy despite xarray patching: https://github.com/pydata/xarray/issues/6524
            [
                np.floor(das_ds["time_idx"]),  # closest time sample below
                np.ceil(das_ds["time_idx"]),  # closest time sample above
            ],
            dim="interp",
        )
    else:
        raise NotImplementedError(
            "Interpolation method not supported: {}".format(method)
        )

    if das_ds["time_idx"].count() == 0:
        raise ValueError(
            "No I/Q time indices correspond to valid measurements within the "
            "image grid."
        )
    if not (
        das_ds.time_idx_round.isel(interp=0).isnull() == das_ds.time_idx_round.isnull()
    ).all():
        raise ValueError(
            "If 1 interpolation weight is set, expected all interpolation"
            "weights to be set."
        )

    # Rotate phase of I/Q signals, based on delays
    das_ds["weights"] = interp_weights * np.exp(
        1j * (2 * np.pi * freq_carrier) * das_ds.tau
    )

    # In case, e.g., "x", "z", and "channel" are not 0-indexed
    # Convert "x", "z", "channel" coords to 0-indices, in case they have been
    # set to something else
    das_ds = das_ds.drop_indexes(das_ds.indexes.keys()).reset_coords(drop=True)

    # Convert from semi-sparse (x, z, channel)-shape array of time indices
    # to list of [x, z, time, channel] indices, and corresponding weights
    # Drop pixels that are not within the receive aperture of given channels
    tmp_dim_order = ("x", "z", "channel", "interp")
    das_ds_flat = das_ds.stack(
        nonzero=tmp_dim_order,
    ).dropna(dim="nonzero", how="all")
    assert (
        das_ds_flat["time_idx_round"].notnull().all()
    ), "Should have dropped null time indices"
    assert das_ds_flat["weights"].notnull().all(), "Should have dropped null weights."
    assert len(das_ds_flat.dims) == 1, "Expected to have stacked all dimensions"
    assert (
        das_ds_flat[["time_idx_round", "weights"]]
        .count()
        .equals(das_ds[["time_idx_round", "weights"]].count())
    ), ".dropna shouldn't change count."

    x_z_multi_indices = np.row_stack((das_ds_flat.x, das_ds_flat.z))
    np.testing.assert_allclose(
        das_ds_flat.time_idx_round,
        das_ds_flat.time_idx_round.astype(int),
        rtol=0,
        atol=1e-6,
        err_msg="Expected time indices to be integers.",
    )
    time_channel_multi_indices = np.row_stack(
        (das_ds_flat.time_idx_round.astype(int), das_ds_flat.channel)
    )

    # Convert from [x, z, time, channel] indices to [x_z, time_channel] indices
    x_z_flat_indices = np.ravel_multi_index(
        tuple(x_z_multi_indices), (width_pixels, depth_pixels)
    )
    time_channel_flat_indices = np.ravel_multi_index(
        tuple(time_channel_multi_indices), (num_time_samples, num_channels)
    )

    # Construct delay-and-sum sparse matrix
    shape = (num_pixels, num_time_samples * num_channels)
    das_matrix = csr_array(
        (das_ds_flat["weights"].values, (x_z_flat_indices, time_channel_flat_indices)),
        shape=shape,
    )

    return das_matrix
