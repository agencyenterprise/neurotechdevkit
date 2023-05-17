import numpy as np
import pytest

import neurotechdevkit as ndk
from neurotechdevkit import scenarios, sources
from neurotechdevkit.results import (
    PulsedResult2D,
    PulsedResult3D,
    SteadyStateResult2D,
    SteadyStateResult3D,
    create_pulsed_result,
    create_steady_state_result,
)
from neurotechdevkit.scenarios import _metrics as metrics


@pytest.fixture
def result_args():
    return dict(
        scenario=None,
        center_frequency=25.0,
        effective_dt=0.005,
        pde="pde",
        shot="shot",
        wavefield=None,
        traces="traces",
    )


@pytest.fixture
def fake_ss_result_2d(result_args):
    return SteadyStateResult2D(**result_args)


@pytest.fixture
def fake_ss_result_3d(result_args):
    return SteadyStateResult3D(**result_args)


@pytest.fixture
def ss_data_2d():
    """Returns wavefield data for 2 space dimensions

    Returns:
        a tuple of (wavefield_data, expected_steady_state), where:
            wavefield_data: 3 dimensional data with steady state waves encoded along the
                time axis.
            expected_steady_state: 2 dimensional expected wave amplitudes
    """
    rng = np.random.default_rng(seed=42)
    expected_steady_state = rng.random((5, 8))
    random_phase = rng.random((5, 8))

    dt = 0.005
    ppp = 8
    freq = 25.0
    t_max = (16 * ppp + 4) * dt
    t = np.arange(0, t_max, dt)
    x = t * freq
    angles = 2 * np.pi * (np.expand_dims(x, (0, 1)) + np.expand_dims(random_phase, 2))
    wavefield_data = np.expand_dims(expected_steady_state, 2) * np.sin(angles)

    return wavefield_data, expected_steady_state


@pytest.fixture
def ss_data_3d():
    """Returns wavefield data for 3 space dimensions

    Returns:
        a tuple of (wavefield_data, expected_steady_state), where:
            wavefield_data: 4 dimensional data with steady state waves encoded along the
                time axis.
            expected_steady_state: 3 dimensional expected wave amplitudes
    """
    rng = np.random.default_rng(seed=43)
    expected_steady_state = rng.random((5, 6, 3))
    random_phase = rng.random((5, 6, 3))

    dt = 0.005
    ppp = 8
    freq = 25.0
    t_max = (16 * ppp + 4) * dt
    t = np.arange(0, t_max, dt)
    x = t * freq
    angles = (
        2 * np.pi * (np.expand_dims(x, (0, 1, 2)) + np.expand_dims(random_phase, 3))
    )
    wavefield_data = np.expand_dims(expected_steady_state, 3) * np.sin(angles)

    return wavefield_data, expected_steady_state


@pytest.fixture
def pulsed_data_2d():
    """Returns wavefield data for 2 space dimensions

    Returns:
        wavefield_data: 3 dimensional data with amplitude waves encoded along the
            time axis.
    """
    rng = np.random.default_rng(seed=42)
    wavefield_data = rng.random(size=(3, 4, 5))
    return wavefield_data


@pytest.fixture
def a_test_scenario_2d():
    """A real 2D scenario that can be saved to disk and reloaded."""
    scenario = scenarios.Scenario1_2D()
    scenario.add_source(
        sources.FocusedSource2D(
            position=np.array([0.02, 0.02]),
            direction=np.array([1.0, -1.0]),
            aperture=0.025,
            focal_length=0.07,
            num_points=100,
        )
    )
    return scenario


@pytest.fixture
def a_test_scenario_3d():
    """A real 3D scenario that can be saved to disk and reloaded."""
    scenario = scenarios.Scenario1_3D()
    scenario.add_source(
        sources.FocusedSource3D(
            position=np.array([0.02, 0.02, 0.0]),
            direction=np.array([1.0, -1.0, 0.0]),
            aperture=0.025,
            focal_length=0.07,
            num_points=100,
        )
    )
    scenario.current_target_id = "target_2"
    return scenario


@pytest.mark.parametrize("by_slice", [False, True])
def test_extract_steady_state_with_2d(fake_ss_result_2d, ss_data_2d, by_slice):
    """Verify that we can correctly extract a 2D steady-state result

    The result should be the same no matter whether the fft is done by slice or not.
    """
    data, expected_steady_state = ss_data_2d
    fake_ss_result_2d.wavefield = data
    measured_steady_state = fake_ss_result_2d._extract_steady_state(by_slice=by_slice)
    np.testing.assert_allclose(measured_steady_state, expected_steady_state)


@pytest.mark.parametrize("by_slice", [False, True])
def test_extract_steady_state_with_3d(fake_ss_result_3d, ss_data_3d, by_slice):
    """Verify that we can correctly extract a 3D steady-state result

    The result should be the same no matter whether the fft is done by slice or not.
    """
    data, expected_steady_state = ss_data_3d
    fake_ss_result_3d.wavefield = data
    print(data.shape, expected_steady_state.shape)
    measured_steady_state = fake_ss_result_3d._extract_steady_state(by_slice=by_slice)
    np.testing.assert_allclose(measured_steady_state, expected_steady_state)


def test_create_steady_state_result_with_2d_wavefield(result_args):
    """Verify that the function returns a SteadyStateResult2D with a 2D wavefield.

    Note that a 2D wavefield has a 3rd axis for time.
    """
    result_args.update({"wavefield": np.ones((4, 5, 10))})
    result = create_steady_state_result(**result_args)
    assert isinstance(result, SteadyStateResult2D)


def test_create_steady_state_result_with_3d_wavefield(result_args):
    """Verify that the function returns a SteadyStateResult3D with a 3D wavefield.

    Note that a 3D wavefield has a 4th axis for time.
    """
    result_args.update({"wavefield": np.ones((4, 5, 6, 10))})
    result = create_steady_state_result(**result_args)
    assert isinstance(result, SteadyStateResult3D)


@pytest.mark.parametrize("n_dims", [2, 5])
def test_create_steady_state_result_with_invalid_wavefield_shape(result_args, n_dims):
    """The function should raise an error if the wavefield is neither 2D nor 3D.

    The error should be raised for either 2 or 5 total dimensions (including 1 for
    time.)
    """
    result_args.update({"wavefield": np.ones((3,) * n_dims)})
    with pytest.raises(ValueError):
        create_steady_state_result(**result_args)


def test_get_steady_state_computes_array(fake_ss_result_2d, ss_data_2d):
    """The steady-state results should be computed if it doesn't already exist."""
    data, _ = ss_data_2d
    fake_ss_result_2d.wavefield = data
    assert fake_ss_result_2d.steady_state is None
    steady_state = fake_ss_result_2d.get_steady_state()
    np.testing.assert_equal(steady_state, fake_ss_result_2d.steady_state)


def test_get_steady_state_does_not_recompute(fake_ss_result_2d):
    """The steady-state results should not be recomputed if it already exists."""
    expected = np.ones((5, 4))
    fake_ss_result_2d.steady_state = expected
    fake_ss_result_2d.get_steady_state()
    assert fake_ss_result_2d.steady_state is expected


def test_metrics_calls_the_appropriate_function(fake_ss_result_2d, monkeypatch):
    """Verify that the expected metrics function is called."""

    def stub(result):
        assert result is fake_ss_result_2d
        return {"a_metric": {"value": 0.42}}

    monkeypatch.setattr(metrics, "calculate_all_metrics", stub)

    fake_ss_result_2d.steady_state = np.ones((5, 4))
    result = fake_ss_result_2d.metrics
    assert result["a_metric"]["value"] == 0.42


def assert_scenario_match(scenario, expected):
    """Check that important scenario parameters match between two scenarios.

    The only parameters tested are those important for saving and reloading results.
    """
    assert scenario.current_target_id == expected.current_target_id
    assert len(scenario.sources) == len(expected.sources)
    for source1, source2 in zip(scenario.sources, expected.sources):
        np.testing.assert_allclose(source1.position, source2.position)
        np.testing.assert_allclose(source1.unit_direction, source2.unit_direction)
        assert source1.aperture == source2.aperture
        assert source1.focal_length == source2.focal_length
        assert source1.num_points == source2.num_points


@pytest.mark.integration
def test_save_round_trip_steady_state_2d(
    fake_ss_result_2d, ss_data_2d, a_test_scenario_2d, tmp_path
):
    """Verify that saving and loading a SteadyStateResult2D reproduces the fields.

    This is an integration test because it saves an actual result object to disk.
    """
    data, expected_steady_state = ss_data_2d
    fake_ss_result_2d.scenario = a_test_scenario_2d
    fake_ss_result_2d.wavefield = data
    fake_ss_result_2d.steady_state = expected_steady_state
    filepath = tmp_path / "result.pkl"
    fake_ss_result_2d.save_to_disk(filepath)
    expected = fake_ss_result_2d
    result = ndk.load_result_from_disk(filepath)

    assert isinstance(result, SteadyStateResult2D)
    np.testing.assert_allclose(result.steady_state, expected.steady_state)
    assert result.center_frequency == expected.center_frequency
    assert result.effective_dt == expected.effective_dt
    assert_scenario_match(result.scenario, expected.scenario)


@pytest.mark.integration
def test_save_round_trip_steady_state_3d(
    fake_ss_result_3d, ss_data_3d, a_test_scenario_3d, tmp_path
):
    """Verify that saving and loading a SteadyStateResult3D reproduces the fields.

    This is an integration test because it saves an actual result object to disk.
    """
    data, expected_steady_state = ss_data_3d
    fake_ss_result_3d.scenario = a_test_scenario_3d
    fake_ss_result_3d.wavefield = data
    fake_ss_result_3d.steady_state = expected_steady_state
    filepath = tmp_path / "result.pkl"
    fake_ss_result_3d.save_to_disk(filepath)
    expected = fake_ss_result_3d
    result = ndk.load_result_from_disk(filepath)

    assert isinstance(result, SteadyStateResult3D)
    np.testing.assert_allclose(result.steady_state, expected.steady_state)
    assert result.center_frequency == expected.center_frequency
    assert result.effective_dt == expected.effective_dt
    assert_scenario_match(result.scenario, expected.scenario)


@pytest.mark.integration
def test_save_round_trip_pulsed_2d(pulsed_data_2d, a_test_scenario_2d, tmp_path):
    """Verify that saving and loading a PulsedResults2D reproduces the fields.

    This is an integration test because it saves an actual result object to disk.
    """
    pulsed_results = PulsedResult2D(
        scenario=a_test_scenario_2d,
        center_frequency=25.0,
        effective_dt=0.005,
        pde="pde",
        wavefield=pulsed_data_2d,
        shot="shot",
        traces=None,
    )

    filepath = tmp_path / "result.pkl"
    pulsed_results.save_to_disk(filepath)
    expected = pulsed_results
    result = ndk.load_result_from_disk(filepath)

    assert isinstance(result, PulsedResult2D)
    np.testing.assert_allclose(result.wavefield, expected.wavefield)
    assert result.center_frequency == expected.center_frequency
    assert result.effective_dt == expected.effective_dt
    assert_scenario_match(result.scenario, expected.scenario)


@pytest.mark.integration
def test_save_round_trip_pulsed_3D(tmp_path, a_test_scenario_3d):
    """Verify that saving and loading a PulsedResults3D reproduces the fields.

    This is an integration test because it saves an actual result object to disk.
    """
    pulsed_results = PulsedResult3D(
        scenario=a_test_scenario_3d,
        center_frequency=25.0,
        effective_dt=0.005,
        pde="pde",
        wavefield=np.arange(0, 40).reshape((2, 2, 2, 5)),
        shot="shot",
        traces=None,
    )

    filepath = tmp_path / "result.pkl"
    pulsed_results.save_to_disk(filepath)
    expected = pulsed_results
    result = ndk.load_result_from_disk(filepath)

    assert isinstance(result, PulsedResult3D)
    np.testing.assert_allclose(result.wavefield, expected.wavefield)
    assert result.center_frequency == expected.center_frequency
    assert result.effective_dt == expected.effective_dt
    assert_scenario_match(result.scenario, expected.scenario)


def test_create_pulsed_result_with_2d_wavefield(result_args):
    """Verify that the function returns a PulsedResult2D with a 2D wavefield.

    Note that a 2D wavefield has a 3rd axis for time.
    """
    result_args.update({"wavefield": np.ones((4, 5, 10))})
    result = create_pulsed_result(**result_args)
    assert isinstance(result, PulsedResult2D)


def test_create_pulsed_result_with_3d_wavefield(result_args):
    """Verify that the function returns a PulsedResult3D with a 3D wavefield.

    Note that a 3D wavefield has a 4th axis for time.
    """
    result_args.update({"wavefield": np.ones((4, 5, 6, 10))})
    result = create_pulsed_result(**result_args)
    assert isinstance(result, PulsedResult3D)


@pytest.mark.parametrize("n_dims", [2, 5])
def test_create_pulsed_result_with_invalid_wavefield_shape(result_args, n_dims):
    """The function should raise an error if the wavefield is neither 2D nor 3D.

    The error should be raised for either 2 or 5 total dimensions (including 1 for
    time.)
    """
    result_args.update({"wavefield": np.ones((3,) * n_dims)})
    with pytest.raises(ValueError):
        create_pulsed_result(**result_args)


def test_pulsed_results_recorded_times(result_args):
    """Verify that recorded timestamps are computed correctly."""
    result_args.update(
        {
            "wavefield": np.zeros(shape=(3, 3, 5)),
            "effective_dt": 2.1,
        }
    )
    result = create_pulsed_result(**result_args)
    timestamps = result._recording_times()
    expected = np.linspace(start=0, stop=8.4, num=5)
    np.testing.assert_allclose(timestamps, expected)


@pytest.mark.parametrize("time_lim", [(0, 0), (1, 0), (0,), (-1, 2), (0, 6)])
def test_validate_time_lim_argument(time_lim, result_args, monkeypatch):
    """Verify that the time_lim validation checks all invalid cases."""

    def mock_times(fake_self):
        return np.array([0, 1, 2, 3, 4, 5])

    monkeypatch.setattr(PulsedResult2D, "_recording_times", mock_times)
    result_args.update({"wavefield": np.ones((4, 5, 6))})
    result = create_pulsed_result(**result_args)
    with pytest.raises(ValueError):
        result._validate_time_lim(time_lim)


def test_raises_error_if_ffmpeg_not_installed(monkeypatch):
    """Verify that raises error if ffmpeg is not installed."""
    import shutil

    def mock_which(command):
        if command == "ffmpeg":
            return None
        else:
            raise ValueError()

    monkeypatch.setattr(shutil, "which", mock_which)
    with pytest.raises(ModuleNotFoundError):
        PulsedResult2D._check_ffmpeg_is_installed()


def test_validate_file_name_error_if_not_mp4():
    """Verify that _validate_file_name checks for file extension."""
    with pytest.raises(ValueError):
        PulsedResult2D._validate_file_name("a_non_mp4_file.fake", False)


def test_validate_file_name_error_if_exists(monkeypatch):
    """Verify that _validate_file_name raises error if file exists."""
    import os

    def mock_file_exists(fake_name):
        if check := fake_name == "a_existent_file.mp4":
            return check
        else:
            raise ValueError()

    monkeypatch.setattr(os.path, "exists", mock_file_exists)
    with pytest.raises(FileExistsError):
        PulsedResult2D._validate_file_name("a_existent_file.mp4", False)


@pytest.mark.parametrize("slice_args", [(1, 0.0), (None, 0.0), (1, None)])
def test_validate_slicing_options(slice_args, result_args):
    """Validates that raises error if trying to slice 2 times the field"""
    result_args.update({"wavefield": np.ones((4, 5, 6, 10))})
    result = create_pulsed_result(**result_args)
    result.recorded_slice = (1, 0.5)
    with pytest.raises(ValueError):
        result._validate_slicing_options(*slice_args)


@pytest.mark.parametrize("recorded_slice", [(1, 0.0), None])
def test_creation_of_pulsed_3d_results_sliced(result_args, recorded_slice):
    """Verify that `PulsedResults3D` can be created regardless of recorded_slice."""
    result_args.update(
        {"recorded_slice": recorded_slice, "wavefield": np.ones((4, 5, 6, 10))}
    )
    result = create_pulsed_result(**result_args)
    assert isinstance(result, PulsedResult3D)
