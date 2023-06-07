import warnings

import pytest

from neurotechdevkit.scenarios._resources import (
    budget_time_and_memory_resources,
    estimate_memory_required,
    estimate_running_time,
)


def test_budget_time_and_memory_resources_warning():
    """Verify that the memory warning is shown."""
    N = 1e4
    with pytest.warns(UserWarning):
        budget_time_and_memory_resources((N, N, N), 1, 1, 1, ram_available_gb=100)


def test_budget_time_and_memory_resources_not_warning():
    """Verify that the warning is not shown when plenty of memory available."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        budget_time_and_memory_resources((10, 10, 10), 1, 2, 3, ram_available_gb=100)


def test_budget_time_and_memory_resources_prints_cpu_time(capsys):
    """Verify that the estimated time is printed to stdout."""
    budget_time_and_memory_resources((200, 300), 1, 2, 3, n_threads=11)
    captured = capsys.readouterr()
    assert len(captured.out) > 0
    assert "Estimated time to complete simulation" in captured.out


@pytest.mark.parametrize(
    "n_threads, n_points, time_steps, expected_time",
    [
        (1, 0, 0, 46.3),
        (11, 1e2, 20, 101.36524218000001),
        (16, 1e3, 30, 47.275800000000004),
        (70, 1e4, 40, 67.54808772),
        (73, 1e9, 50, 38227.758102149994),
    ],
)
def test_estimate_running_time_steady_state(
    n_threads, n_points, time_steps, expected_time
):
    """Verify that the estimated time is calculated correctly."""
    warnings.simplefilter("ignore")
    result = estimate_running_time(
        n_points=n_points, time_steps=time_steps, n_threads=n_threads, is_pulsed=False
    )
    assert result == expected_time


@pytest.mark.parametrize(
    "n_threads, n_points, time_steps, simulated_not_recorded_frames, expected_time",
    [
        (24, 58594371, 2722, 1361, 1820.8734061823275),
        (16, 58594371, 5682, 5562, 2212.1376088474767),
        (16, 153791, 4754, 4634, 626.7545025354104),
        (8, 4791321, 2731, 2671, 299.5110687863803),
    ],
)
def test_estimate_running_time_pulsed(
    n_threads, n_points, time_steps, simulated_not_recorded_frames, expected_time
):
    """Verify that the estimated time is calculated correctly."""
    warnings.simplefilter("ignore")
    result = estimate_running_time(
        n_points=n_points,
        time_steps=time_steps,
        simulated_not_recorded_frames=simulated_not_recorded_frames,
        n_threads=n_threads,
        is_pulsed=True,
    )
    assert result == expected_time


def test_estimate_running_time_raises_warning():
    """Verify that warning is raised when capping number of threads."""
    with pytest.warns(UserWarning):
        estimate_running_time(1e5, 10, n_threads=1000)


@pytest.mark.parametrize(
    "n_points, time_undersampling, n_cycles_steady_st, time_steps, expected_prediction",
    [
        (0, 0, 0, 0, 7),
        (1e10, 10, 20, 1e3, 1500),
        (1e8, 10, 20, 1e3, 17),
    ],
)
def test_estimate_memory_required_steady_state_returns_value(
    n_points, time_undersampling, n_cycles_steady_st, time_steps, expected_prediction
):
    """Verify that estimated memory is inline with intuition."""
    estimated_memory = estimate_memory_required(
        n_points,
        time_undersampling,
        time_steps=time_steps,
        is_pulsed=False,
        n_cycles_steady_state=n_cycles_steady_st,
    )
    assert int(estimated_memory) == expected_prediction


@pytest.mark.parametrize(
    "n_points, not_recorded_frames, recording_time_undersampling, expected_prediction",
    [
        (58594371, 1361, 1361.0, 747),
        (58594371, 2041, 680.5, 353),
        (4791321, 623, 623.0, 46),
        (4791321, 934, 311.5, 33),
    ],
)
def test_estimate_memory_required_pulsed_returns_value(
    n_points,
    not_recorded_frames,
    recording_time_undersampling,
    expected_prediction,
):
    """Verify that estimated memory is inline with intuition."""
    estimated_memory = estimate_memory_required(
        n_points=n_points,
        recording_time_undersampling=recording_time_undersampling,
        simulated_not_recorded_frames=not_recorded_frames,
        is_pulsed=True,
    )
    assert int(estimated_memory) == expected_prediction
