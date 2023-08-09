import os
import warnings
from typing import Optional

import numpy as np
import numpy.typing as npt
import psutil


def get_available_ram_memory() -> float:
    """Return the available RAM memory in GB.

    Returns:
        The available RAM memory in GB.
    """
    avail_ram = psutil.virtual_memory().total / 10**9
    return avail_ram


def get_available_cpus() -> int:
    """Return the available CPUs.

    Returns:
        The available number of CPUs.
    """
    if avail_cpus := os.cpu_count() is None:
        return 0

    return avail_cpus


def estimate_memory_required(
    n_points: int,
    recording_time_undersampling: int,
    is_pulsed: bool = False,
    time_steps: Optional[int] = None,
    n_cycles_steady_state: Optional[int] = None,
    simulated_not_recorded_frames: Optional[int] = None,
) -> int:
    """Estimate the RAM memory required in GB to run a steady state simulation.

    A linear model is used to estimate the memory required. The coefficients of the
    model were determined by least squares regression on benchmark simulations.

    Estimations are approximate (within 20% of the true value). Memory estimations
    below 4 GB are rounded up to 4GB.

    Args:
        n_points: the total number of nodes in the grid.
        recording_time_undersampling: the number of skipped frames while recording.
        is_pulsed: whether the simulation is pulsed or steady state.
        time_steps: the number of time steps of the simulation.
        n_cycles_steady_state: the number of cycles in steady state.
        simulated_not_recorded_frames: the number of frames simulated but not recorded.

    Returns:
        The estimate of RAM (in GB) required to run the simulation.
    """
    MIN_MEMORY_USAGE = {"pulsed": 4, "steady": 4}
    SIMULATION_TYPE = "pulsed" if is_pulsed else "steady"
    intercept = {"steady": 7.400, "pulsed": 3.56416648}
    coefs = {
        "steady": np.array([1.49829208e-10, -3.29276854e00, 1.38685890e00]),
        "pulsed": np.array(
            [5.33930321e-08, -6.19314285e-04, 4.79853449e-04, 1.14050083e-01]
        ),
    }
    if is_pulsed:
        assert simulated_not_recorded_frames is not None
        vals = np.array(
            [
                n_points,
                simulated_not_recorded_frames,
                recording_time_undersampling,
                is_pulsed,
            ]
        )
    else:
        assert time_steps is not None
        assert n_cycles_steady_state is not None
        vals = np.array(
            [n_points * time_steps, recording_time_undersampling, n_cycles_steady_state]
        )

    predicted_memory = intercept[SIMULATION_TYPE] + np.dot(coefs[SIMULATION_TYPE], vals)
    if is_pulsed:
        predicted_memory = np.exp(predicted_memory)

    return np.clip(predicted_memory, MIN_MEMORY_USAGE[SIMULATION_TYPE], None)


def estimate_running_time(
    n_points: int,
    time_steps: int,
    n_threads: int,
    is_pulsed: bool = False,
    simulated_not_recorded_frames: Optional[int] = None,
) -> float:
    """Estimate the time (in seconds) to complete the simulation.

    Computation time is estimated from a linear model. Which linear model to select
    is determined by the number of CPU/threads available. In the case that number of
    threads isn't precomputed the estimation will be done with the closest number of
    threads.

    The maximum number of threads accepted is 64. Values would be capped to 64 to
    avoid tricky extrapolations.

    Args:
        n_points: the total number of nodes in the simulation grid.
        time_steps: the number of time steps in the simulation.
        n_threads: the number of threads to use during the simulation. Note that
        currently is all available CPUs.
        is_pulsed: whether the simulation is pulsed or steady state.
        simulated_not_recorded_frames: the number of frames simulated but not recorded.
    """
    MAX_NUMBER_OF_THREADS = {"pulsed": 24, "steady": 64}
    MIN_EXECUTION_TIME = {"pulsed": 120, "steady": 10}
    SIMULATION_TYPE = "pulsed" if is_pulsed else "steady"

    # Don't allow for extrapolation
    if n_threads > MAX_NUMBER_OF_THREADS[SIMULATION_TYPE]:
        warnings.warn(
            message=(
                "Time estimation is done with the maximum of "
                f"{MAX_NUMBER_OF_THREADS[SIMULATION_TYPE]},"
                f" instead of the available {n_threads} threads."
            ),
            category=UserWarning,
        )
        n_threads = MAX_NUMBER_OF_THREADS[SIMULATION_TYPE]

    models = {
        "pulsed": {
            2: {
                "coefs": np.array([1.13015289e-04, 3.18889017e00, -3.83404541e00]),
                "intercept": 0.0,
            },
            4: {
                "coefs": np.array([5.06337500e-05, 1.20934791e00, -1.30082292e00]),
                "intercept": 0.0,
            },
            8: {
                "coefs": np.array([3.61267043e-05, 5.72646770e-01, -5.38181167e-01]),
                "intercept": 0.0,
            },
            16: {
                "coefs": np.array([2.50914277e-05, 2.37813632e-01, -1.09553375e-01]),
                "intercept": 0.0,
            },
            24: {
                "coefs": np.array([2.15004025e-05, 2.68392830e-01, -1.24536692e-01]),
                "intercept": 0.0,
            },
        },
        "steady": {
            1: {"coefs": [0.000217, -0.00266], "intercept": 46.3},
            4: {"coefs": [1.18667e-04, -2.39815e-05], "intercept": 84.1},
            8: {"coefs": [7.13638e-05, -2.09471e-03], "intercept": 101.4},
            16: {"coefs": [0.000366, -0.00634], "intercept": 47.1},
            32: {"coefs": [3.89005e-05, -7.96062e-04], "intercept": 82.4},
            64: {"coefs": [3.81606e-05, -8.37957e-04], "intercept": 67.2},
        },
    }

    threads_avail = np.array(list(models[SIMULATION_TYPE].keys()))
    closest = np.argmin(np.abs(n_threads - threads_avail))

    lr = models[SIMULATION_TYPE][threads_avail[closest]]
    if is_pulsed:
        vals = np.array([n_points, time_steps, simulated_not_recorded_frames])
    else:
        vals = np.array([n_points, time_steps])
    coefs = np.array(lr["coefs"])

    estimated_time = lr["intercept"] + np.dot(coefs, vals)
    return max(MIN_EXECUTION_TIME[SIMULATION_TYPE], estimated_time)


def budget_time_and_memory_resources(
    grid_shape: npt.NDArray[np.int_],
    recording_time_undersampling: int,
    time_steps: int,
    is_pulsed: bool,
    n_cycles_steady_state: Optional[int] = None,
    n_threads: int = get_available_cpus(),
    ram_available_gb: float = get_available_ram_memory(),
) -> None:
    """Inform the user of the time and memory resources needed for the simulation.

    The default value for n_threads assumes that all CPUs in the computer are used.
    The default value for ram_available_gb assumes that all RAM memory in the computer
    is available for the simulation.

    The function prints a message estimating the time required to complete the
    computation and the memory required.
    In the case that the memory required is larger than the available memory, it warns
    the user but doesn't interrupt execution.

    Args:
        grid_shape: the dimensions of the simulation grid.
        recording_time_undersampling: the number of skipped frames while recording.
        time_steps: the number of time steps in the simulation.
        is_pulsed: whether the simulation is pulsed or steady state.
        n_cycles_steady_state: the number of cycles in steady state.
        n_threads: the number of threads to use during the simulation (default is is all
            available CPUs).
        ram_available_gb: the RAM memory available for the simulation (default is all
            available RAM memory).
    """
    n_points = int(np.prod(grid_shape))
    simulated_not_recorded_frames = None
    if is_pulsed:
        simulated_not_recorded_frames = time_steps - recording_time_undersampling

    # Memory estimation
    ram_required_gb = estimate_memory_required(
        n_points,
        recording_time_undersampling,
        is_pulsed,
        time_steps,
        n_cycles_steady_state=n_cycles_steady_state,
        simulated_not_recorded_frames=simulated_not_recorded_frames,
    )
    if ram_required_gb >= ram_available_gb:
        warnings.warn(
            """The simulation might run out of memory:
            Estimated RAM required : {r} GB, available {a} GB.
            """.format(
                r=ram_required_gb, a=round(ram_available_gb)
            ),
            category=UserWarning,
        )

    # Running time estimation
    estimated_time = estimate_running_time(
        n_points,
        time_steps,
        n_threads,
        is_pulsed,
        simulated_not_recorded_frames=simulated_not_recorded_frames,
    )
    if estimated_time < 60:
        unit = "seconds"
        estimated_time = int(estimated_time)
    else:
        estimated_time = int(estimated_time / 60.0)
        unit = "minutes"

    print(
        f"Estimated time to complete simulation: {estimated_time} {unit}."
        f" Memory required is {ram_required_gb} GB (available {ram_available_gb} GB)."
        " These values are approximated."
    )
