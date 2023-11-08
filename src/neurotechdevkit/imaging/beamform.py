"""Beam-form I/Q signals for ultrasound imaging."""

from enum import IntEnum, unique
from typing import Optional, Tuple

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
        x: 2-D array specifying the x-coordinates of the [z, x] image grid.
        z: 2-D array specifying the z-coordinates of the [z, x] image grid.
        freq_sampling: sampling frequency of the input signals.
        freq_carrier: center/carrier frequency of the input signals.
        **kwargs: additional arguments passed to `delay_and_sum_matrix`

    Returns:
        Beamformed signals at the specified [z, x] image grid.

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
    assert x.ndim == 2, "Expected x to have shape (depth_pixels, width_pixels)."
    assert z.ndim == 2, "Expected z to have shape (depth_pixels, width_pixels)."
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
     Calculate the delay-and-sum sparse matrix for beamforming I/Q signals.

    Args:
        num_time_samples: Number of time samples in the I/Q signals.
        num_channels: Number of channels/elements in the transducer array.
        x: x-coordinates (width) of the image grid (shape: depth_pixels x width_pixels).
        z: z-coordinates (depth) of the image grid (shape: depth_pixels x width_pixels).
        pitch: Center-to-center distance between two adjacent elements/channels (m).
        tx_delays: Transmit delays (seconds) for each channel (length: num_channels).
        freq_sampling: Sampling frequency of the I/Q signals (Hz).
        freq_carrier: Carrier/center frequency of the I/Q signals (Hz).
        start_time: Start time of the I/Q signal (s). Default is 0.
        speed_sound: Speed of sound in the medium (m/s). Default is 1540.
        f_number: Receive f-number or focal ratio.
            If None, it will be estimated based on width and bandwidth.
        width: Width of each channel/element in the array (m).
            Required if f_number is None.
        bandwidth: Fractional bandwidth at -6dB.
            Required if f_number is None.
        method: Interpolation method across time dimension.

    Returns:
        delay-and-sum matrix. Shape: (num_pixels, num_samples*num_channels)

    Assumptions:
        - The input signal is in-phase/quadrature (I/Q) signals.
        - The input signal is sampled at a constant rate.
        - The sensor array is linear.
        - The sensor array is parallel to the x-axis and centered at (0, 0).

    Implementation notes:
        - Alternatively, delay-and-sum can be implemented using an explicit
            for-loop over the delays. However, Python for-loops are much
            slower than  optimized sparse matrix multiplication.
        - Memory usage for the delay-and-sum matrix is:
            O(num_pixels * num_channels * num_interpolation_points).
        - We use the `xarray` library to simplify broadcasting and indexing.

    For details on delay-and-sum by matrix-multiplication:
        https://doi.org/10.1016/j.ultras.2020.106309
    """
    # Check inputs
    assert num_time_samples > 0, "Number of time samples must be positive."
    assert num_channels > 0, "Number of channels/elements must be positive."
    assert (x.size > 0) and (z.size > 0), "Number of pixels must be positive."
    assert x.ndim == 2, "Expected x to have shape (depth_pixels, width_pixels)."
    assert z.ndim == 2, "Expected z to have shape (depth_pixels, width_pixels)."
    assert x.shape == z.shape, "Expected image grid x and z to have the same shape."
    assert len(tx_delays) == num_channels, "Expected one transmit-delay per channel."
    if f_number is None:
        assert width is not None, "Element width is required to estimate f-number."
        assert (
            bandwidth is not None
        ), "Element bandwidth is required to estimate f-number."
        f_number = _optimize_f_number(
            element_width=width,
            bandwidth_fractional=bandwidth,
            freq_carrier=freq_carrier,
            speed_sound=speed_sound,
        )
    assert isinstance(f_number, float)
    assert f_number >= 0, "f-number must be non-negative."

    # Helper parameters
    depth_pixels, width_pixels = x.shape

    # Convert to xarray for more intuitive broadcasting
    x_dataarray = xr.DataArray(x, dims=("z", "x"))  # keep as 0-index
    z_dataarray = xr.DataArray(z, dims=("z", "x"))
    tx_delays_dataarray = xr.DataArray(tx_delays, dims=("channel",))

    # assume transducer centered on (0, 0) parallel to x-axis
    x_channels = xr.DataArray(
        (np.arange(num_channels) - (num_channels - 1) / 2) * pitch, dims=("channel",)
    )  # keep as 0-index
    z_channels = xr.DataArray(np.zeros(x_channels.shape), dims=("channel",))

    # Calculate time-of-flight/propagation times, from start to receive.
    tau = _calculate_time_of_flight(
        x_dataarray=x_dataarray,
        z_dataarray=z_dataarray,
        x_channels=x_channels,
        z_channels=z_channels,
        tx_delays_dataarray=tx_delays_dataarray,
        speed_sound=speed_sound,
        depth_pixels=depth_pixels,
        width_pixels=width_pixels,
        num_channels=num_channels,
    )

    # Convert wave propagation time to sample index (fast-time)
    # Keep `tau` for later use
    das_ds = tau.to_dataset(name="tau")
    das_ds["time_idx"] = (tau - start_time) * freq_sampling
    assert das_ds.sizes == {
        "z": depth_pixels,
        "x": width_pixels,
        "channel": num_channels,
    }, "Expected one fast-time index and weight per pixel/channel combo."

    # Each channel only needs to consider pixels within its receive aperture
    # (i.e., angle-of-view = 2*alpha)
    receptive_field = _calculate_receptive_fields(
        f_number=f_number,
        x_dataarray=x_dataarray,
        z_dataarray=z_dataarray,
        x_channels=x_channels,
        depth_pixels=depth_pixels,
        width_pixels=width_pixels,
        num_channels=num_channels,
    )
    das_ds = das_ds.where(receptive_field)

    # We have discrete time samples, which do not correspond exactly to the
    # pixel grid. Interpolate to find the pixel grid values
    (
        interp_weights,
        das_ds["time_idx_round"],
        is_valid_time_idx,
    ) = _interpolate_onto_pixel_grid(
        time_idx_float=das_ds["time_idx"],
        max_time_samples=num_time_samples,
        method=method,
    )
    das_ds = das_ds.where(is_valid_time_idx)

    # Check for potential underlying errors
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

    das_matrix = _construct_delay_and_sum_matrix(
        das_ds=das_ds,
        width_pixels=width_pixels,
        depth_pixels=depth_pixels,
        num_time_samples=num_time_samples,
        num_channels=num_channels,
    )

    return das_matrix


def _optimize_f_number(
    element_width: float,
    bandwidth_fractional: float,
    freq_carrier: float,  # Hz
    speed_sound: float,  # m/s
) -> float:
    """Optimize f-number based on other parameters.

    Args:
        element_width: width of each element/channel in the array (m)
        bandwidth_fractional: fractional bandwidth at -6dB
        freq_carrier: center/carrier frequency of the input signals (Hz)
        speed_sound: speed of sound (m/s)

    Returns:
        f-number, also known as the focal ratio or f-stop
    """
    assert (
        0 < bandwidth_fractional < 2
    ), "Fractional bandwidth at -6dB must be in (0, 2)."

    # Eq. 14 of https://doi.org/10.1016/j.ultras.2020.106309
    wavelength_min = speed_sound / (freq_carrier * (1 + bandwidth_fractional / 2))

    optimal_directivity = 0.71

    # Eq. 13 of https://doi.org/10.1016/j.ultras.2020.106309
    result = minimize_scalar(
        lambda theta: np.abs(
            _directivity(theta, element_width, wavelength_min) - optimal_directivity
        ),
        bounds=(0, np.pi / 2),
    )
    alpha = result.x
    f_number = 1 / (2 * np.tan(alpha))

    return f_number


def _directivity(theta, element_width, wavelength):
    """Calculate directivity of a single element/channel.

    Eq. 11 of https://doi.org/10.1016/j.ultras.2020.106309
    "For a piston-like element in a soft baffle"

    Args:
        theta: receive_angle (radians)
        element_width: width of each element/channel in the array (m)
        wavelength: sound wavelength (m)

    Returns:
        directivity of a single element/channel
    """
    return np.cos(theta) * np.sinc(element_width / wavelength * np.sin(theta))


def _calculate_time_of_flight(
    x_dataarray: xr.DataArray,
    z_dataarray: xr.DataArray,
    x_channels: xr.DataArray,
    z_channels: xr.DataArray,
    tx_delays_dataarray: xr.DataArray,
    speed_sound: float,
    depth_pixels: int,
    width_pixels: int,
    num_channels: int,
) -> xr.DataArray:
    """Calculate time of flight (including transmit delays) for each pixel and channel.

    "Time-of-flight" refers to the time it takes for an ultrasound wave to
    travel from the transducer element to a specific point in the imaging field
    and then back to the transducer after being reflected by an object.

    Args:
        x_dataarray: x-coordinates (width) of the image grid.
        z_dataarray: z-coordinates (depth) of the image grid.
        x_channels: x-coordinates (width) of the transducer elements.
        z_channels: z-coordinates (depth) of the transducer elements.
        tx_delays_dataarray: transmit delays (seconds) for each channel.
        speed_sound: speed of sound in the medium (m/s).
        depth_pixels: number of pixels in the depth (z) dimension.
            Used for validating array sizes.
        width_pixels: number of pixels in the width (x) dimension.
            Used for validating array sizes.
        num_channels: number of channels/elements in the transducer array.
            Used for validating array sizes.

    Returns:
        time of flight (seconds) for each pixel and channel.
    """
    # Effective transmit-distance from each element to each pixel
    tx_distance = tx_delays_dataarray * speed_sound + np.sqrt(
        (x_dataarray - x_channels) ** 2 + (z_dataarray - z_channels) ** 2
    )
    assert tx_distance.sizes == {
        "z": depth_pixels,
        "x": width_pixels,
        "channel": num_channels,
    }, "Expected one transmit distance per pixel and channel."
    # transmit-pulse reaches pixel, based on the "closest" transducer channel/element
    tx_distance = tx_distance.min(dim="channel")
    assert tx_distance.sizes == {
        "z": depth_pixels,
        "x": width_pixels,
    }, "Expected transmit delays to be calculated for each pixel."

    # Receive-delay:
    # Calculate distances from each pixel to each receive-element/channel
    rx_distance = np.sqrt(
        (x_dataarray - x_channels) ** 2 + (z_dataarray - z_channels) ** 2
    )
    assert rx_distance.sizes == {  # type: ignore
        # https://github.com/pydata/xarray/issues/6524
        "z": depth_pixels,
        "x": width_pixels,
        "channel": num_channels,
    }, "Expected one receive delay per pixel and channel."

    tau = (tx_distance + rx_distance) / speed_sound
    assert tau.sizes == {  # type: ignore
        # https://github.com/pydata/xarray/issues/6524
        "z": depth_pixels,
        "x": width_pixels,
        "channel": num_channels,
    }, "Expected one time-of-flight per pixel and channel."

    return tau


def _calculate_receptive_fields(
    f_number: float,
    x_dataarray: xr.DataArray,
    z_dataarray: xr.DataArray,
    x_channels: xr.DataArray,
    depth_pixels: float,
    width_pixels: float,
    num_channels: float,
) -> xr.DataArray:
    """Calculate which pixels are within the receptive field of each channel.

    Args:
        f_number: receive f-number or focal ratio.
        x_dataarray: x-coordinates (width) of the image grid.
        z_dataarray: z-coordinates (depth) of the image grid.
        x_channels: x-coordinates (width) of the transducer elements.
        depth_pixels: number of pixels in the depth (z) dimension.
            Used for validating array sizes.
        width_pixels: number of pixels in the width (x) dimension.
            Used for validating array sizes.
        num_channels: number of channels/elements in the transducer array.
            Used for validating array sizes.

    Returns:
        Boolean DataArray indicating whether each pixel is within the receptive
        field of each channel.
    """
    if f_number == 0:
        # All pixels are within receptive field
        is_within_aperture = xr.ones_like(x_dataarray, dtype=bool)
        is_within_aperture = is_within_aperture.broadcast_like(x_channels)

    else:
        assert f_number > 0, "f-number must be non-negative."
        # assume linear array at z=0
        half_aperture = z_dataarray / (2 * f_number)
        is_within_aperture = (  # type: ignore
            np.abs(x_dataarray - x_channels) <= half_aperture  # type: ignore
        )
        # https://github.com/pydata/xarray/issues/6524

    assert is_within_aperture.sizes == {  # type: ignore
        # https://github.com/pydata/xarray/issues/6524
        "z": depth_pixels,
        "x": width_pixels,
        "channel": num_channels,
    }, "Expected to know whether each pixel was within each channel's aperture."

    return is_within_aperture


def _interpolate_onto_pixel_grid(
    time_idx_float: xr.DataArray,
    max_time_samples: int,
    method: InterpolationMethod,
) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Calculate interpolation weights for delay-and-sum matrix.

    We have discrete time samples, which do not correspond exactly to the pixel
    grid. This function interpolates to find the values along the pixel grid

    Arguments:
        time_idx_float: time indices (float) that don't fall on exact time grid
        max_time_samples: maximum number of time samples in the I/Q signals
        method: interpolation method across time dimension

    Returns:
        - interp_weights: interpolation weights for new `interp` dimension
        - time_idx_round: rounded (int) time indices
        - is_valid_time_idx: boolean mask whether each time index is valid
    """
    if method == InterpolationMethod.NEAREST:
        interp_weights = xr.DataArray([1], dims=("interp",))
        is_valid_time_idx = (time_idx_float >= 0) & (
            time_idx_float <= (max_time_samples - 1)
        )
        time_idx_round = time_idx_float.round()
        # Add dimension to match other interpolation methods
        time_idx_round = time_idx_round.expand_dims("interp")
    elif method == InterpolationMethod.LINEAR:
        # E.g., a time index of 2.3 gets 0.7*pixel_2 + 0.3*pixel_3
        is_valid_time_idx = (time_idx_float >= 0) & (
            time_idx_float <= (max_time_samples - 1)
        )
        # Weighted sum of two nearest time samples
        interp_weights = xr.concat(
            [
                1 - (time_idx_float % 1),  # how close to upper time sample?
                time_idx_float % 1,  # how close to lower time sample?
            ],
            dim="interp",
        )
        time_idx_round = xr.concat(  # type: ignore
            # https://github.com/pydata/xarray/issues/6524
            [
                np.floor(time_idx_float),  # closest time sample below
                np.ceil(time_idx_float),  # closest time sample above
            ],
            dim="interp",
        )
    else:
        raise NotImplementedError(
            "Interpolation method not supported: {}".format(method)
        )

    return interp_weights, time_idx_round, is_valid_time_idx


def _construct_delay_and_sum_matrix(
    das_ds: xr.Dataset,
    *,
    width_pixels: int,
    depth_pixels: int,
    num_time_samples: int,
    num_channels: int,
) -> csr_array:
    """Convert Dataset to sparse matrix for delay-and-sum beamforming.

    Handles a lot of the bookkeeping for converting from a semi-dense array to a
    sparse matrix.

    Args:
        das_ds: Dataset containing the following variables:
            - time_idx_round: time indices (int)
            - weights: interpolation weights along `interp` dimension
        width_pixels: number of pixels in the width (x) dimension.
        depth_pixels: number of pixels in the depth (z) dimension.
        num_time_samples: number of time samples in the I/Q signals.
        num_channels: number of channels/elements in the transducer array.

    Returns:
        delay-and-sum sparse matrix. Shape: (num_pixels, num_samples*num_channels)
    """
    # Convert "x", "z", "channel" coords to 0-indices, in case they have been
    # set to something else
    das_ds = das_ds.drop_indexes(das_ds.indexes.keys()).reset_coords(drop=True)

    # Convert from semi-sparse (z, x, channel)-shape array of time indices
    # to list of [z, x, time, channel] indices, and corresponding weights
    # Drop pixels that are not within the receive aperture of given channels
    tmp_dim_order = ("z", "x", "channel", "interp")
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

    # Convert to multi-indices
    z_x_multi_indices = np.row_stack((das_ds_flat.z, das_ds_flat.x))
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
    z_x_flat_indices = np.ravel_multi_index(
        tuple(z_x_multi_indices), (depth_pixels, width_pixels)
    )
    time_channel_flat_indices = np.ravel_multi_index(
        tuple(time_channel_multi_indices), (num_time_samples, num_channels)
    )

    # Construct delay-and-sum sparse matrix
    shape = (depth_pixels * width_pixels, num_time_samples * num_channels)
    das_matrix = csr_array(
        (das_ds_flat["weights"].values, (z_x_flat_indices, time_channel_flat_indices)),
        shape=shape,
    )

    return das_matrix
