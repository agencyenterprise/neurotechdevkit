import os
import warnings

import numpy as np
import numpy.typing as npt
import psutil


def get_available_ram_memory() -> float:
    """Returns the available RAM memory in GB.

    Returns:
        The available RAM memory in GB.
    """
    avail_ram = psutil.virtual_memory().total / 10**9
    return avail_ram


def get_available_cpus() -> int:
    """Returns the available CPUs.

    Returns:
        The available number of CPUs.
    """
    if avail_cpus := os.cpu_count() is None:
        return 0

    return avail_cpus


def estimate_memory_required(
    n_points: int,
    recording_time_undersampling: int,
    n_cycles_steady_state: int,
    time_steps: int,
) -> int:
    """Estimates the RAM memory required in GB to run a steady state simulation.

    A linear model is used to estimate the memory required. The coefficients of the
    model were determined by least squares regression on benchmark simulations.

    Estimations are approximate (within 20% of the true value). Memory estimations
    below 4 GB are rounded up to 4GB.

    Args:
        n_points: the total number of nodes in the grid.
        recording_time_undersampling: the number of skipped frames while recording.
        n_cycles_steady_state: the number of cycles in steady state.
        time_steps: the number of time steps of the simulation.

    Returns:
        The estimate of RAM (in GB) required to run the simulation.
    """

    intercept = 7.400
    coefs = np.array([1.49829208e-10, -3.29276854e00, 1.38685890e00])
    vals = np.array(
        [n_points * time_steps, recording_time_undersampling, n_cycles_steady_state]
    )

    predicted_memory = int(intercept + np.dot(coefs, vals))

    return np.clip(predicted_memory, 4, None)


def estimate_running_time(
    n_points: int,
    time_steps: int,
    n_threads: int,
) -> float:
    """Estimates the time (in seconds) to complete the simulation.

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
    """
    MAX_NUMBER_OF_THREADS = 64

    # Don't allow for extrapolation
    if n_threads > MAX_NUMBER_OF_THREADS:
        warnings.warn(
            f"Time estimation is done with the maximum of {MAX_NUMBER_OF_THREADS},"
            f" instead of the available {n_threads} threads.",
            category=UserWarning,
        )
        n_threads = MAX_NUMBER_OF_THREADS

    models = {
        1: {"coefs": [0.000217, -0.00266], "intercept": 46.3},
        4: {"coefs": [1.18667e-04, -2.39815e-05], "intercept": 84.1},
        8: {"coefs": [7.13638e-05, -2.09471e-03], "intercept": 101.4},
        16: {"coefs": [0.000366, -0.00634], "intercept": 47.1},
        32: {"coefs": [3.89005e-05, -7.96062e-04], "intercept": 82.4},
        64: {"coefs": [3.81606e-05, -8.37957e-04], "intercept": 67.2},
    }

    threads_avail = np.array(list(models.keys()))
    closest = np.argmin(np.abs(n_threads - threads_avail))

    lr = models[threads_avail[closest]]
    vals = np.array([n_points, time_steps])
    coefs = np.array(lr["coefs"])
    return lr["intercept"] + np.dot(coefs, vals)


def budget_time_and_memory_resources(
    grid_shape: npt.NDArray[np.int_],
    recording_time_undersampling: int,
    n_cycles_steady_state: int,
    time_steps: int,
    n_threads: int = get_available_cpus(),
    ram_available_gb: float = get_available_ram_memory(),
) -> None:
    """
    Informs the user of the time and memory resources needed to complete the simulation.

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
        n_cycles_steady_state: the number of cycles in steady state.
        time_steps: the number of time steps in the simulation.
        n_threads: the number of threads to use during the simulation (default is is all
            available CPUs).
        ram_available_gb: the RAM memory available for the simulation (default is all
            available RAM memory).
    """

    n_points = int(np.prod(grid_shape))

    # Memory estimation
    ram_required_gb = estimate_memory_required(
        n_points, recording_time_undersampling, n_cycles_steady_state, time_steps
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
    estimated_time = estimate_running_time(n_points, time_steps, n_threads)
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
