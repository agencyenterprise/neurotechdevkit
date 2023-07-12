"""Beam-form I/Q signals for ultrasound imaging."""

from enum import IntEnum
from typing import Optional, Tuple, Dict

import numpy as np
from scipy.interpolate import griddata, interp1d
from scipy.sparse import csr_array
import xarray as xr


class InterpolationMethod(IntEnum):
    """Interpolation methods for beamforming.

    Value corresponds to the number of points used for interpolation.
    """
    NEAREST = 1  # nearest-neighbor interpolation
    LINEAR = 2
    QUADRATIC = 3
    LANZCOS_3 = 3  # 3-lobe Lanczos interpolation
    LSQ_PARABOLA_5 = 5  # 5-point least-squares parabolic interpolation
    LANZCOS_5 = 6 # 5-lobe Lanczos interpolation


def beamform_delay_and_sum(
    SIG: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    *das_matrix_args,
    interp_method: InterpolationMethod = InterpolationMethod.NEAREST,
) -> np.ndarray:
    """Delay-And-Sum beamforming.

    Currently, assumes that the input signal is in-phase/quadrature (I/Q) signals.

    Parameters:
        SIG: 2-D complex-valued array containing I/Q signals. Shape: (time_samples, num_channels)
            channels usually correspond to transducer elements.
            Note: Can also stack a 3rd dimension for repeated-echos (slow-time)
        x: 2-D array specifying the x-coordinates of the [x, z] image grid.
        z: 2-D array specifying the z-coordinates of the [x, z] image grid.
        *das_matrix_args: additional arguments for the delay-and-sum beamforming.
        interp_method: interpolation method for delay-and-sum matrix.

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
    assert SIG.ndim == 2, "Expected SIG to have shape (time_samples, num_channels)."
    assert np.iscomplexobj(SIG), "Expected SIG to be complex-valued I/Q signals."
    assert x.ndim == 2, "Expected x to have shape (width_pixels, depth_pixels)."
    assert z.ndim == 2, "Expected z to have shape (width_pixels, depth_pixels)."
    assert x.shape == z.shape, "Expected image grid x and z to have the same shape."
    num_time_samples, num_channels = SIG.shape

    # Check for potential underlying errors before allocating memory
    delay_and_sum_matrix([nl, nc], x[0], z[0], *args)

    # Chunking
    # Large data can generate tall DAS matrices. The input data (X,Z) are
    # chunked to avoid out-of-memory issues.

    # Maximum possible array bytes
    MPAB = np.finfo(np.float64).max

    # The number of bytes required to store a sparse matrix M is roughly:
    #     bytes = 16*nnz(M) + 8*(size(M,2)+1)
    # (for a 64-bit system)
    #
    # In our case:
    # nnz(M) < (number of transducer elements)*...
    #          (number of interpolating points)*...
    #          (number of grid points)
    # size(M,2) = min(number of RF/IQ samples,number of grid points)
    #
    # Roughly:
    # bytes < 16*(number of transducer elements)*...
    #            (number of interpolating points)*...
    #            (number of grid points)
    #

    # Number of chunks
    NoE = nc  # number of elements
    bytes = 16 * NoE * Npoints * np.prod(x.shape)
    factor = 20  # other large variables in DASMTX + ...
    # compromise for-loops vs. memory
    # (arbitrary value > 4-5)
    Nchunks = np.ceil(factor * bytes / MPAB)

    # Delay-and-Sum
    SIG = SIG.reshape(nl * nc, -1)
    bfSIG = np.zeros([np.prod(siz0), SIG.shape[1]], dtype=SIG.dtype)

    idx = np.round(np.linspace(0, np.prod(siz0), Nchunks + 1)).astype(int)

    for k in range(Nchunks):
        # DAS matrices using DASMTX
        M, param = delay_and_sum_matrix((~isIQ + isIQ * 1j) * np.array([nl, nc]),
                          x[idx[k]:idx[k + 1]],
                          z[idx[k]:idx[k + 1]],
                          *args)

        # Delay-and-Sum
        bfSIG[idx[k]:idx[k + 1], :] = M @ SIG

    bfSIG = bfSIG.reshape([*siz0, SIG.shape[1]])

    return bfSIG


def delay_and_sum_matrix(
    num_time_samples: int,
    num_channels: int,
    x: np.ndarray,
    z: np.ndarray,
    *,
    pitch: float,  # m  center-to-center distance between two adjacent elements/channels
    delaysTX: np.ndarray,
    fs: float,  # Hz
    fc: float,  # Hz
    t0: float = 0,
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
    assert len(delaysTX) == num_channels, "Expected one transmit-delay per channel."
    if f_number is None:
        # Optimize f-number based on other parameters
        assert width is not None, "Element width is required to estimate f-number."
        assert bandwidth is not None, "Element bandwidth is required to estimate f-number."
        assert 0 < bandwidth < 2, "Fractional bandwidth at -6dB must be in (0, 2)."
    else:
        assert f_number >= 0, 'f-number must be non-negative.'
    if method != InterpolationMethod.NEAREST:
        raise NotImplementedError("Only nearest-neighbor interpolation is supported.")

    # Helper parameters
    width_pixels, depth_pixels = x.shape
    num_pixels = x.size

    # Convert to xarray for more intuitive broadcasting
    x = xr.DataArray(x, dims=("x", "z"))  # keep as 0-index
    z = xr.DataArray(z, dims=("x", "z"))
    delaysTX = xr.DataArray(delaysTX, dims=("channel",))

    # transducer centered on (0, 0) parallel to x-axis
    xe = (np.arange(num_channels) - (num_channels - 1) / 2) * pitch
    xe = xr.DataArray(xe, dims=("channel",))  # keep as 0-index
    ze = np.zeros(xe.shape)
    ze = xr.DataArray(ze, dims=("channel",))

    # TODO: interpolate transducers for better estimation of delays
    # For now, don't interpolate
    xTi = xe
    zTi = ze

    # Effective transmit-distance from each (interpolated) element to each pixel
    # TODO: add new dimension / broadcast for the channel dimension
    dTX = delaysTX * c + np.sqrt((x - xTi) ** 2 + (z - zTi) ** 2)
    assert dTX.sizes == {
        "x": width_pixels, "z": depth_pixels, "channel": num_channels,
    }, "Expected one transmit distance per pixel and channel."
    # transmit-pulse reaches pixel, based on the "closest" transducer channel/element
    dTX = dTX.min(dim="channel")
    assert dTX.sizes == {
        "x": width_pixels, "z": depth_pixels,
    }, "Expected transmit delays to be calculated for each pixel."

    # Receive-delay:
    # Calculate distances from each pixel to each receive-element/channel
    dRX = np.sqrt((x - xe)**2 + (z - ze)**2)
    assert dRX.sizes == {
        "x": width_pixels, "z": depth_pixels, "channel": num_channels,
    }, "Expected one receive delay per pixel and channel."

    # Wave propagation times
    tau = (dTX + dRX) / c
    # # Clear large variables
    # del dTX
    # del dRX

    # Convert wave propagation time to sample index (fast-time)
    time_idx = (tau - t0) * fs
    assert time_idx.sizes == {
        "x": width_pixels, "z": depth_pixels, "channel": num_channels,
    }, "Expected one fast-time index per pixel/channel combo."

    # Each channel only needs to consider pixels within its receive aperture
    # (i.e., angle-of-view = 2*alpha)
    if f_number > 0:
        # assume linear array at z=0
        half_aperture = z / (2 * f_number)
        is_within_aperture = np.abs(x - xe) <= half_aperture
        assert is_within_aperture.sizes == {
            "x": width_pixels, "z": depth_pixels, "channel": num_channels,
        }, "Expected to know whether each pixel was within each channel's aperture."
        time_idx = time_idx.where(is_within_aperture)
        # # Clear large variable
        # del is_within_aperture
    else:
        assert f_number == 0, "f-number must be non-negative."

    # Use nearest-neighbor interpolation
    assert method == InterpolationMethod.NEAREST
    num_interp_points = method.value
    time_idx = time_idx.round()
    interp_weights = xr.DataArray([1], dims=("interp",))
    is_valid_time_idx = (time_idx >= 0) & (time_idx < num_time_samples)

    if time_idx.where(is_valid_time_idx).count() == 0:
        raise ValueError(
            "No I/Q time indices correspond to valid measurements within the image grid."
        )
    time_idx = time_idx.where(is_valid_time_idx)

    # Rotate phase of I/Q signals, based on delays
    tau = tau.where(time_idx.notnull())
    das_weights = interp_weights * np.exp(1j * (2 * np.pi * fc) * tau)
    # For nearest-neighbor,
    # we can drop interp dimension simply because we only have one interp point
    das_weights = das_weights.squeeze("interp")

    # In case, e.g., "x", "z", and "channel" are not 0-indexed
    # Convert "x", "z", "channel" coords to 0-indices, in case they have been
    # set to something else
    time_idx = time_idx.drop_indexes(time_idx.indexes.keys()).reset_coords(drop=True)

    # Convert from semi-sparse (x, z, channel)-shape array of time indices
    # to list of [x, z, time, channel] indices
    time_idx_flat = time_idx.stack(
        nonzero=("x", "z", "channel"),
    ).dropna(dim="nonzero")
    assert time_idx_flat.ndim == 1, "Expected to have stacked all dimensions"
    assert time_idx_flat.count() == time_idx.count(), "Stacking shouldn't change count."
    # Convert rounded-float time_indices back to int indices
    np.testing.assert_allclose(time_idx_flat, time_idx_flat.astype(int), rtol=0, atol=1e-3)
    time_idx_flat = time_idx_flat.astype(int)
    x_z_multi_indices = np.column_stack((time_idx_flat.x, time_idx_flat.z))
    time_channel_multi_indices = np.column_stack((time_idx_flat.values, time_idx_flat.channel))

    # Convert from [x, z, time, channel] indices to [x_z, time_channel] indices
    x_z_flat_indices = np.ravel_multi_index(
        x_z_multi_indices, (width_pixels, depth_pixels)
    )
    time_channel_flat_indices = np.ravel_multi_index(
        time_channel_multi_indices, (num_time_samples, num_channels)
    )

    # Construct delay-and-sum sparse matrix
    shape = (width_pixels * depth_pixels, num_time_samples * num_channels)
    assert das_weights_flat.size == time_idx.count()
    das_matrix = csr_array(
        (das_weights_flat, (x_z_flat_indices, time_channel_flat_indices)), shape=shape
    )

    return das_matrix
