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
    "n_threads, n_points, time_steps",
    [(1, 0, 0), (11, 1e2, 20), (16, 1e3, 30), (70, 1e4, 40), (73, 1e9, 50)],
)
def test_estimate_running_time_stress(n_threads, n_points, time_steps):
    """Verify that the estimated time is a positive float."""
    warnings.simplefilter("ignore")
    result = estimate_running_time(n_points, n_threads, time_steps)
    assert isinstance(result, float)
    assert result > 0


def test_estimate_running_time_raises_warning():
    """Verify that warning is raised when capping number of threads."""
    with pytest.warns(UserWarning):
        estimate_running_time(1e5, 10, n_threads=1000)


def test_estimate_memory_required_returns_value():
    """Verify that estimated memory is inline with intuition."""
    assert estimate_memory_required(0, 0, 0, 0) > 0
    assert estimate_memory_required(1e10, 10, 20, 1e3) > 1000
    assert estimate_memory_required(1e8, 10, 20, 1e3) > 10
