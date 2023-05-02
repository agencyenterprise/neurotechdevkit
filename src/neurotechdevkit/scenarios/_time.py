from __future__ import annotations

from typing import Mapping

import numpy as np
import stride
from frozenlist import FrozenList
from mosaic.types import Struct


def select_simulation_time_for_steady_state(
    *,
    grid: stride.Grid,
    materials: Mapping[str, Struct],
    freq_hz: float,
    time_to_steady_state: float | None,
    n_cycles_steady_state: int,
    delay: float,
) -> stride.Time:
    """Determines how much time (in seconds) to simulate for a steady-state simulation.

    In order to reach near-steady-state conditions, we need the simulation to run for
    enough time for the waves to fully propagate to all edges of the simulation and
    complete most reflections. Therefore, we select the simulation time to be equal to
    the amount of time it would take for a wave to travel from the one corner of the
    simulation space to the opposite corner and back if the scenario was filled with
    whatever material has the lowest speed of sound.

    Args:
        grid: the `Grid` for the scenario.
        materials: a dict between material names and the material properties.
        freq_hz: the center frequency (in hertz) of the source used in the scenario.
        time_to_steady_state: Specify the amount of time (in seconds) the simulation
            should run before measuring the steady-state amplitude. If the value is
            None, this time will automatically be set to the amount of time it would
            take to propagate from one corner to the opposite and back in the medium
            with the slowest speed of sound in the scenario.
        n_cycles_steady_state: The number of complete cycles to use when calculating
            the steady-state wave amplitudes.
        delay: the maximum delay (in seconds) in the sources to account for.

    Returns:
        The amount of time (in seconds) to simulate.
    """
    if time_to_steady_state is None:
        min_speed_of_sound = min([m.vp for m in materials.values()])
        diagonal_length = np.linalg.norm(grid.space.size)
        time_to_steady_state = (2 * diagonal_length) / min_speed_of_sound

    period = 1.0 / freq_hz
    sim_time = delay + time_to_steady_state + n_cycles_steady_state * period
    return sim_time


def select_simulation_time_for_pulsed(
    *,
    grid: stride.Grid,
    materials: Mapping[str, Struct],
    delay: float,
):
    """Determines how much time (in seconds) to simulate for a pulsed simulation.

    For pulsed simulations, we usually want to simulate enough time for the wave to
    fully propagate to the edges of the scenario, but we don't need to worry about
    reaching a steady-state. Therefore, we select the simulation time to be equal to the
    amount of time it would take for a wave to travel from the one corner of the
    simulation space to the opposite corner if the scenario was filled with whatever
    material has the lowest speed of sound.

    Args:
        grid: the `Grid` for the scenario.
        materials: a dict between material names and the material properties.
        delay: the maximum delay (in seconds) in the sources to account for.

    Returns:
        The amount of time (in seconds) to simulate.
    """
    min_speed_of_sound = min([m.vp for m in materials.values()])
    diagonal_length = np.linalg.norm(grid.space.size)
    return delay + diagonal_length / min_speed_of_sound


def create_time_grid(*, freq_hz: float, ppp: int, sim_time: float) -> stride.Time:
    """Build the stride `Time` object for the simulation.

    `Time` parameters depend on simulation parameters and so needs to be rebuilt anytime
    the simulation parameters change.

    Args:
        freq_hz: the center frequency (in hertz) of the source used in the scenario.
        ppp: the number of time points per cycle to use when calculating dt.
        sim_time: the amount of time (in seconds) to simulate.

    Returns:
        The prepared `Time` object.
    """
    t_min = 0.0
    period = 1.0 / freq_hz
    dt = period / ppp
    return stride.Time(start=t_min, stop=sim_time, step=dt)


def find_largest_delay_in_sources(sources: FrozenList) -> float:
    """Finds the largest delay (in seconds) among all sources

    Args:
        sources: a list with all sources in a scenario.

    Returns:
        The maximum time delay (in seconds) among all sources
    """
    if len(sources) == 0:
        return 0.0

    return np.max([s.point_source_delays.max() for s in sources])
