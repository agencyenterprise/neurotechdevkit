from __future__ import annotations

import numpy as np
import numpy.typing as npt

from . import _results as results


def calculate_all_metrics(
    result: results.SteadyStateResult,
) -> dict[str, dict[str, float | str]]:
    """Calculate all metrics for the steady-state result and return as a dictionary.

    The keys for the dictionary are the names of the metrics. The value for each metric
    is another dictionary containing the following:
        value: The value of the metric.
        unit-of-measurement: The unit of measurement for the metric.
        description: A text description of the metric.

    Args:
        result: the object containing the steady-state simulation results.

    Returns:
        The dictionary structure containing metrics and their descriptions.
    """
    return {
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
            for layer in result.scenario.ordered_layers
        },
    }


def calculate_mechanical_index(
    result: results.SteadyStateResult, layer: str | None = None
) -> float:
    """Calculate the mechanical index of the simulation result for a particular layer.

    The mechanical index is the peak negative pressure divided by the square root of the
    center frequency.

    Args:
        result: The Result object containing the simulation results.
        layer: The layer within which to calculate the mechanical index. If None, the
            default, the mechanical index is calculated over the entire simulation
            space.

    Returns:
        The mechanical index (in Pa √s̅)
    """
    if layer is None:
        mask = np.ones_like(result.get_steady_state(), dtype=bool)
    else:
        mask = result.scenario.get_layer_mask(layer)
    ss_amp_in_brain: npt.NDArray[np.float_] = np.ma.masked_array(
        result.get_steady_state(), mask=~mask
    )
    peak_neg_pressure = np.max(ss_amp_in_brain)
    return peak_neg_pressure / np.sqrt(result.center_frequency)


def calculate_focal_gain(result: results.SteadyStateResult) -> float:
    """Calculate the focal gain of the simulation result.

    The focal gain is taken to be the ratio between the steady-state amplitude within
    the target region divided by the steady-state amplitude within the rest of the
    brain, presented in Decibels.

    Args:
        result: The Result object containing the simulation results.

    Returns:
        The focal gain (in dB)
    """
    target_mask = result.scenario.get_target_mask()
    brain_mask = result.scenario.get_layer_mask("brain")

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
        result: The Result object containing the simulation results.

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
        result: The Result object containing the simulation results.

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
    """Calculate the time-averaged intensity within the brain but outside of the target
    region.

    The time-averaged intensity is equal to the integral (over time) of the pressure
    amplitude times 1/(rho*vp).

    For a steady-state result with a sinusoidal source wavelet, the integral over a
    single cycle is A**2 / (2 * f) where A is the steady-state pressure amplitude and f
    is the center frequency.

    The intensity is averaged over the brain region excluding the target region.

    Args:
        result: The Result object containing the simulation results.

    Returns:
        the time-averaged intensity averaged over the brain but outside of the target
            region (in W/m^2).
    """
    target_mask = result.scenario.get_target_mask()
    brain_mask = result.scenario.get_layer_mask("brain")

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
        result: The Result object containing the simulation results.

    Returns:
        the pulse-averaged intensity averaged over the target region (in W/m^2).
    """
    return calculate_i_ta_target(result)


def calculate_i_pa_off_target(result: results.SteadyStateResult) -> float:
    """Calculate the pulse-averaged intensity within the brain but outside of the target
    region.

    For steady-state results, the pulse-averaged intensity is equal to the time-average
    intensity.

    See `calculate_i_ta_off_target` for more information about how the metric is
    calculated.

    Args:
        result: The Result object containing the simulation results.

    Returns:
        the pulse-averaged intensity averaged over the brain but outside of the target
            region (in W/m^2).
    """
    return calculate_i_ta_off_target(result)


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
            value: The value to be converted.

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
