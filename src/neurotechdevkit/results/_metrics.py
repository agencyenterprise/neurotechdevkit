from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import scipy.ndimage

from . import _results as results


def calculate_all_metrics(
    result: results.SteadyStateResult,
) -> dict[str, dict[str, float | int | str]]:
    """Calculate all metrics for the steady-state result and return as a dictionary.

    The keys for the dictionary are the names of the metrics. The value for each metric
    is another dictionary containing the following:

    - value: the value of the metric.
    - unit-of-measurement: the unit of measurement for the metric.
    - description: A text description of the metric.

    Args:
        result: the object containing the steady-state simulation results.

    Returns:
        The dictionary structure containing metrics and their descriptions.
    """
    return {
        "focal_pressure": {
            "value": calculate_focal_pressure(result, layer="brain"),
            "unit-of-measurement": "Pa",
            "description": ("The peak pressure amplitude within the brain."),
        },
        "focal_volume": {
            "value": calculate_focal_volume(result, layer="brain"),
            "unit-of-measurement": "voxels",
            "description": (
                "The region of the brain where the pressure amplitude is above"
                " 50%% of the maximum pressure amplitude. If more than one"
                " region is above 50%% of the maximum pressure amplitude, the"
                " volume of largest connected region is returned. Also called"
                " the -6 dB focal volume."
            ),
        },
        "focal_gain": {
            "value": calculate_focal_gain(result),
            "unit-of-measurement": "dB",
            "description": (
                "The ratio between the mean steady-state pressure amplitude inside"
                " the target and that of the ambient, expressed in Decibels. The"
                " ambient pressure is calculated by the mean steady-state pressure"
                " amplitude inside the brain but excluding the target."
            ),
        },
        **{
            f"FWHM_{axis}": {
                "value": calculate_focal_fwhm(result, axis=axis, layer="brain"),
                "unit-of-measurement": "grid steps",
                "description": (
                    "The full-width at half-maximum-pressure (FWHM) along the"
                    f" {axis}-axis. The FWHM is calculated by taking a 1-D"
                    " profile through the focal point. Also called the -6 dB"
                    " focal width."
                ),
            }
            for axis in ("x", "y", "z")[: result.get_steady_state().ndim]
        },
        "I_ta_target": {
            "value": Conversions.convert(
                from_uom="W/m²", to_uom="mW/cm²", value=calculate_i_ta_target(result)
            ),
            "unit-of-measurement": "mW/cm²",
            "description": (
                "The temporal-average intensity (ITA) within the target region. ITA is"
                " calculated by integrating the wave intensity over a full period and"
                " dividing by the period. An example FDA recommended limit for ITA is"
                " 720 mW/cm² (the exact value can vary depending on the intended use)."
            ),
        },
        "I_ta_off_target": {
            "value": Conversions.convert(
                from_uom="W/m²",
                to_uom="mW/cm²",
                value=calculate_i_ta_off_target(result),
            ),
            "unit-of-measurement": "mW/cm²",
            "description": (
                "The temporal-average intensity (ITA) within the brain but outside of"
                " the target. ITA is calculated by integrating the wave intensity over"
                " a full period and dividing by the period. An example FDA recommended"
                " limit for ITA is 720 mW/cm² (the exact value can vary depending on"
                " the intended use)."
            ),
        },
        "I_pa_target": {
            "value": Conversions.convert(
                from_uom="W/m²", to_uom="W/cm²", value=calculate_i_pa_target(result)
            ),
            "unit-of-measurement": "W/cm²",
            "description": (
                "The pulse-average intensity (IPA) within the target region. IPA is"
                " calculated by integrating the intensity over the pulse and dividing"
                " by the length of the pulse. For steady-state waves, IPA is equal to"
                " the temporal-average intensity. An example FDA recommended limit for"
                " IPA is 190 W/cm² (the exact value can vary depending on the intended"
                " use)."
            ),
        },
        "I_pa_off_target": {
            "value": Conversions.convert(
                from_uom="W/m²", to_uom="W/cm²", value=calculate_i_pa_off_target(result)
            ),
            "unit-of-measurement": "W/cm²",
            "description": (
                "The pulse-average intensity (IPA) within the brain but outside of the"
                " target. IPA is calculated by integrating the intensity over the"
                " pulse and dividing by the length of the pulse. For steady-state"
                " waves, IPA is equal to the temporal-average intensity. An example"
                " FDA recommended limit for IPA is 190 W/cm² (the exact value can vary"
                " depending on the intended use)."
            ),
        },
        "mechanical_index_all": {
            "value": calculate_mechanical_index(result, layer=None),
            "unit-of-measurement": "Pa √s̅",
            "description": (
                "The mechanical index (MI) over all materials in the full simulation"
                " volume. MI is defined as peak negative pressure divided by the"
                " square root of the frequency of the ultrasound wave. An example"
                " FDA recommended limit for MI is 1.9 (the exact value can vary"
                " depending on the intended use)."
            ),
        },
        **{
            f"mechanical_index_{layer}": {
                "value": calculate_mechanical_index(result, layer=layer),
                "unit-of-measurement": "Pa √s̅",
                "description": (
                    f"The mechanical index (MI) within the {layer} layer. The MI is"
                    " defined as peak negative pressure divided by the square root of"
                    " the frequency of the ultrasound wave. An example FDA recommended"
                    " limit for MI is 1.9 (the exact value can vary depending on the"
                    " intended use)."
                ),
            }
            for layer in result.scenario.material_layers
        },
    }


def calculate_focal_pressure(
    result: results.SteadyStateResult, layer: str | None = None
) -> float:
    """Calculate the focal pressure of the simulation result for a particular layer.

    Focal pressure is also known as peak pressure.

    Args:
        result: the Result object containing the simulation results.
        layer: the layer within which to calculate the focal pressure. If None, the
            default, the focal pressure is calculated over the entire simulation
            space.

    Returns:
        The focal pressure (in Pa)
    """
    ss_amp_masked = _get_steady_state_in_layer(result, layer=layer)
    focal_pressure = np.max(ss_amp_masked)
    return focal_pressure


def calculate_focal_position(
    result: results.SteadyStateResult, layer: str | None = None
) -> Tuple[np.int_, ...]:
    """Calculate the focal position of the simulation result for a particular layer.

    The focal position is the position of peak pressure.

    Args:
        result: the Result object containing the simulation results.
        layer: the layer within which to calculate the focal position. If None, the
            default, the focal position is calculated over the entire simulation
            space.

    Returns:
        The focal position (as grid index tuple)
    """
    ss_amp_masked = _get_steady_state_in_layer(result, layer=layer)
    focal_position = np.unravel_index(
        np.argmax(ss_amp_masked, axis=None), ss_amp_masked.shape
    )
    return focal_position


def calculate_mechanical_index(
    result: results.SteadyStateResult, layer: str | None = None
) -> float:
    """Calculate the mechanical index of the simulation result for a particular layer.

    The mechanical index is the peak negative pressure divided by the square root of the
    center frequency.

    Args:
        result: the Result object containing the simulation results.
        layer: the layer within which to calculate the mechanical index. If None, the
            default, the mechanical index is calculated over the entire simulation
            space.

    Returns:
        The mechanical index (in Pa √s̅)
    """
    ss_amp_masked = _get_steady_state_in_layer(result, layer=layer)
    peak_neg_pressure = np.max(ss_amp_masked)
    return peak_neg_pressure / np.sqrt(result.center_frequency)


def calculate_focal_gain(result: results.SteadyStateResult) -> float:
    """Calculate the focal gain of the simulation result.

    The focal gain is taken to be the ratio between the steady-state amplitude within
    the target region divided by the steady-state amplitude within the rest of the
    brain, presented in Decibels.

    Args:
        result: the Result object containing the simulation results.

    Returns:
        The focal gain (in dB)
    """
    target_mask = result.scenario.get_target_mask()
    brain_mask = result.scenario.material_masks["brain"]

    ss_in_target: npt.NDArray[np.float_] = np.ma.masked_array(
        result.get_steady_state(), mask=~target_mask
    )
    ss_brain_excl_target: npt.NDArray[np.float_] = np.ma.masked_array(
        result.get_steady_state(), mask=(~brain_mask | target_mask)
    )
    amp_ratio = np.mean(ss_in_target) / np.mean(ss_brain_excl_target)
    return 10 * np.log10(amp_ratio)


def calculate_i_ta(result: results.SteadyStateResult) -> npt.NDArray[np.float_]:
    """Calculate the time-averaged intensity for a steady-state result.

    The time-averaged intensity is equal to the integral (over time) of the pressure
    amplitude times 1/(rho*vp).

    For a steady-state result with a sinusoidal source wavelet, the integral over a
    single cycle is A**2 / (2 * f) where A is the steady-state pressure amplitude and f
    is the center frequency.

    Args:
        result: the Result object containing the simulation results.

    Returns:
        the time-averaged intensity at each point in space (in W/m^2).
    """
    freq = result.center_frequency
    rho = result.scenario.get_field_data("rho")
    vp = result.scenario.get_field_data("vp")
    return result.get_steady_state() ** 2 / (2 * freq * rho * vp)


def calculate_i_ta_target(result: results.SteadyStateResult) -> float:
    """Calculate the time-averaged intensity within the target region.

    The time-averaged intensity is equal to the integral (over time) of the pressure
    amplitude times 1/(rho*vp).

    For a steady-state result with a sinusoidal source wavelet, the integral over a
    single cycle is A**2 / (2 * f) where A is the steady-state pressure amplitude and f
    is the center frequency.

    Args:
        result: the Result object containing the simulation results.

    Returns:
        the time-averaged intensity averaged over the target region (in W/m^2).
    """
    target_mask = result.scenario.get_target_mask()
    i_spta = calculate_i_ta(result)
    i_spta_in_target: npt.NDArray[np.float_] = np.ma.masked_array(
        i_spta, mask=~target_mask, dtype=float
    )
    return float(np.mean(i_spta_in_target))


def calculate_i_ta_off_target(result: results.SteadyStateResult) -> float:
    """Calculate the time-averaged intensity.

    The time-averaged intensity is calculated within the brain but outside of the
    target region.

    The time-averaged intensity is equal to the integral (over time) of the pressure
    amplitude times 1/(rho*vp).

    For a steady-state result with a sinusoidal source wavelet, the integral over a
    single cycle is A**2 / (2 * f) where A is the steady-state pressure amplitude and f
    is the center frequency.

    The intensity is averaged over the brain region excluding the target region.

    Args:
        result: the Result object containing the simulation results.

    Returns:
        the time-averaged intensity averaged over the brain but outside of the target
            region (in W/m^2).
    """
    target_mask = result.scenario.get_target_mask()
    brain_mask = result.scenario.material_masks["brain"]

    i_spta = calculate_i_ta(result)
    i_spta_outside_target: npt.NDArray[np.float_] = np.ma.masked_array(
        i_spta, mask=(~brain_mask | target_mask)
    )
    return float(np.mean(i_spta_outside_target))


def calculate_i_pa_target(result: results.SteadyStateResult) -> float:
    """Calculate the pulse-averaged intensity within the target region.

    For steady-state results, the pulse-averaged intensity is equal to the time-average
    intensity.

    See `calculate_i_ta_target` for more information about how the metric is calculated.

    Args:
        result: the Result object containing the simulation results.

    Returns:
        the pulse-averaged intensity averaged over the target region (in W/m^2).
    """
    return calculate_i_ta_target(result)


def calculate_i_pa_off_target(result: results.SteadyStateResult) -> float:
    """Calculate the pulse-averaged intensity.

    The pulse-averaged intensity is calculated within the brain but outside of the
    target region.

    For steady-state results, the pulse-averaged intensity is equal to the time-average
    intensity.

    See `calculate_i_ta_off_target` for more information about how the metric is
    calculated.

    Args:
        result: the Result object containing the simulation results.

    Returns:
        the pulse-averaged intensity averaged over the brain but outside of the target
            region (in W/m^2).
    """
    return calculate_i_ta_off_target(result)


def calculate_focal_volume(
    result: results.SteadyStateResult,
    layer: str | None = "brain",
) -> float:
    """Calculate the focal volume of the simulation result.

    From https://doi.org/10.1121/10.0013426:
    > The focal volume is calculated by thresholding the pressure field inside
    > the brain to 50% of the maximum value and then counting the voxels in the
    > largest connected component.

    Args:
        result: the Result object containing the simulation results.
        layer: the layer within which to calculate the focal volume.
            If None, the focal volume is calculated over the entire grid.

    Returns:
        The focal volume (in voxels).
    """
    ss_amp_brain_masked = _get_steady_state_in_layer(result, layer=layer)
    above_threshold = ss_amp_brain_masked >= (0.5 * ss_amp_brain_masked.max())
    # `scipy.ndimage.label` expects a normal numpy array, so let's fill the masked array
    if isinstance(above_threshold, np.ma.MaskedArray):
        above_threshold = above_threshold.filled(False)
    # Get contiguous regions
    labeled_mask, _ = scipy.ndimage.label(above_threshold)
    # Count the occurrences of each label in the labeled array
    unique_labels, label_counts = np.unique(labeled_mask, return_counts=True)
    # Exclude background, which `scipy.ndimage.label` labels as 0
    label_counts = label_counts[unique_labels != 0]
    assert label_counts.sum(), "Expected >=1 connected regions above 50% threshold"
    # Get size of largest connected component
    return label_counts.max()


def calculate_focal_fwhm(
    result: results.SteadyStateResult,
    *,
    axis: str | int,
    layer: str | None = "brain",
) -> float:
    """Calculate the full-width at half-maximum-pressure (FWHM) along an axis.

    Uses the 1-D profile through the focal position.

    Args:
        result: the Result object containing the simulation results.
        axis: the axis along which to calculate the FWHM. Can be specified as a string
            ("x", "y", or "z") or as an integer (0, 1, or 2).
        layer: the layer within which to calculate the FWHM
            If None, the FWHM is calculated over the entire grid.

    Returns:
        The full-width at half-maximum (in integer grid steps).
    """
    if isinstance(axis, str):
        # Convert axis to index
        axis = ["x", "y", "z"].index(axis.lower())

    # Extract 1-D slice going through the focal position along `axis`
    focal_position_idx = calculate_focal_position(result, layer=layer)
    ss_amp_masked = _get_steady_state_in_layer(result, layer=layer)
    slicer: List[np.int_ | slice] = list(focal_position_idx)
    slicer[axis] = slice(None)
    focal_slice = ss_amp_masked[tuple(slicer)]

    # Find which indices are above the half-maximum-pressure
    focal_position_axis_idx = focal_position_idx[axis]
    above_half_max = focal_slice >= (0.5 * focal_slice[focal_position_axis_idx])

    # `scipy.ndimage.label` expects a normal numpy array, so let's fill the masked array
    if isinstance(above_half_max, np.ma.MaskedArray):
        above_half_max = above_half_max.filled(False)
    labeled_mask, _ = scipy.ndimage.label(above_half_max)
    # We only care about the connected region containing the focal position
    fwhm_group_label = labeled_mask[focal_position_axis_idx]
    fwhm = np.sum(labeled_mask == fwhm_group_label)

    return fwhm


class Conversions:
    """A set of unit-of-measurement conversion tools."""

    @staticmethod
    def convert(from_uom, to_uom, value):
        """Convert a value from one unit-of-measurement to another.

        Conversions are requested using specific strings for the starting and desired
        unit-of-measurement. The reason for this is that attempts to specify function
        names that indicate a single conversion have resulted in overly verbose or
        difficult names. Units-of-measurement can be written much more succinctly in a
        string than in a function name, which results in much more readable function
        usage with strings.

        Currently supported conversions:
        * W/m² to mW/cm²
        * W/m² to W/cm²

        If an unsupported conversion is requested, an exception will be raised.

        Args:
            from_uom: A string containing unit-of-measurement from which to convert.
            to_uom: A string containing unit-of-measurement to which to convert.
            value: the value to be converted.

        Raises:
            ValueError: If an unsupported conversion is requested.

        Returns:
            The converted value.
        """
        if from_uom == "W/m²" and to_uom == "mW/cm²":
            return value * 0.1

        if from_uom == "W/m²" and to_uom == "W/cm²":
            return value * 1e-4

        raise ValueError(f"Unsupported conversion: from '{from_uom}' to '{to_uom}'")


def _get_steady_state_in_layer(
    result: results.SteadyStateResult,
    layer: str | None = None,
) -> npt.NDArray | np.ma.MaskedArray:
    """Get the steady-state pressure amplitude within a particular layer.

    Args:
        result: the Result object containing the simulation results.
        layer: the layer to extract

    Returns:
        The steady-state pressure amplitude, with non-layer values masked.
    """
    steady_state = result.get_steady_state()
    if layer is None:
        return steady_state
    else:
        return np.ma.masked_array(
            steady_state,
            mask=~result.scenario.material_masks[layer],
        )
