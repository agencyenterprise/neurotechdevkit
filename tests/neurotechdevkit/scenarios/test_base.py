from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest
import stride

from neurotechdevkit.grid import Grid
from neurotechdevkit.problem import Problem
from neurotechdevkit.results import PulsedResult, SteadyStateResult
from neurotechdevkit.scenarios._base import Scenario
from neurotechdevkit.scenarios._utils import wavelet_helper
from neurotechdevkit.sources import FocusedSource3D, PlanarSource3D, Source


class ScenarioBaseTester(Scenario):
    """A class which can be used to test attributes and methods of Scenario."""

    material_properties = {}
    material_masks = {
        "brain": np.zeros((20, 30, 40), dtype=bool),
        "skin": np.zeros((20, 30, 40), dtype=bool),
    }

    sources = [
        PlanarSource3D(
            position=np.array([0.05, 0.1, 0.05]),
            direction=np.array([0.0, 0.0, 1.0]),
            aperture=0.05,
            num_points=123,
        )
    ]
    origin = np.array([-0.1, -0.1, 0.0])

    def __init__(self):
        self.center_frequency = 5e5
        self._make_grid()
        self.problem = self._compile_problem()

    def _make_grid(self):
        extent = np.array([2.0, 3.0, 4.0])
        grid = Grid.make_grid(extent=extent, dx=0.1)
        self.grid = grid

    def _compile_problem(self) -> Problem:
        problem = Problem(grid=self.grid)
        problem.medium.add(stride.ScalarField(name="vp", grid=self.grid))
        problem.medium.add(stride.ScalarField(name="rho", grid=self.grid))
        problem.medium.add(stride.ScalarField(name="alpha", grid=self.grid))
        # problem.medium.add(stride.ScalarField(name="layer", grid=self.grid))
        return problem

    def get_default_source(self) -> Source:
        return self.default_source

    def get_target_mask(self) -> npt.NDArray[np.bool_]:
        return np.zeros_like(self.shape).astype(bool)


@pytest.fixture
def base_tester():
    """Returns an instance of ScenarioBaseTester."""
    return ScenarioBaseTester()


@pytest.fixture
def tester_with_time(base_tester):
    """Returns a ScenarioBaseTester that has time information defined."""
    time = stride.Time(start=0.0, stop=0.5, step=0.01)
    base_tester.problem.grid.time = time
    return base_tester


@pytest.fixture
def tester_with_fixed_extra_space(base_tester):
    """Returns a ScenarioBaseTester that has time information defined."""
    base_tester.problem.space.shape = (10, 10, 10)
    base_tester.problem.space.extra = (0, 0, 0)
    return base_tester


@pytest.fixture
def a_source():
    """Returns a Source instance that can be used for testing."""
    return FocusedSource3D(
        position=np.array([0.0, 0.0, 0.1]),
        direction=np.array([1.0, 1.0, 0.0]),
        aperture=0.05,
        focal_length=0.1,
        num_points=321,
    )


@pytest.fixture
def fake_pde():
    """Returns a fake PDE object that can be used for testing."""

    class FakePDE:
        """A Fake PDE."""

        def __init__(self):
            self.return_value = "I'll be back"
            self.last_args = None
            self.last_kwargs = None
            fake_wf = np.arange(12).reshape((3, 4, 1))
            self.wavefield = SimpleNamespace(data=fake_wf)

        async def __call__(self, *args, **kwargs):
            self.last_args = args
            self.last_kwargs = kwargs
            return self.return_value

    return FakePDE()


@pytest.fixture
def tester_with_layers(base_tester):
    """Returns a Scenario for testing which includes layer masks."""
    layer_field = base_tester.problem.medium.fields["layer"]
    layer_field.data[:5] = 0
    layer_field.data[5:] = 1
    return base_tester


@pytest.mark.parametrize("sim_mode", ["steady-state", "pulsed"])
def test_setup_shot_sources_locations(tester_with_time, a_source, sim_mode):
    """The shot should add the sources to the problem and itself.

    This is tested here for a single source. Multiple sources are tested in
    test_shot.py.
    """
    assert tester_with_time.problem.geometry.num_locations == 0
    shot = tester_with_time._setup_shot(
        [a_source], freq_hz=50.0, simulation_mode=sim_mode
    )
    assert tester_with_time.problem.geometry.num_locations == a_source.num_points
    shot_locations = shot.sources
    problem_locations = tester_with_time.problem.geometry.locations
    assert shot_locations == problem_locations


@pytest.mark.parametrize("sim_mode", ["steady-state", "pulsed"])
def test_setup_shot_attributes(tester_with_time, a_source, sim_mode):
    """Verify some of the attributes of the shot match expectations"""
    shot = tester_with_time._setup_shot(
        [a_source], freq_hz=50.0, simulation_mode=sim_mode
    )
    assert shot.id == 0
    assert shot.problem == tester_with_time.problem
    assert shot.geometry == tester_with_time.problem.geometry
    assert len(shot.receivers) == 0


def test_setup_shot_wavelet_for_steady_state(tester_with_time, a_source):
    """_setup_shot should create a wavelet for the steady-state shot.

    This is tested here for a single source. Multiple sources are tested in
    test_shot.py.
    """
    shot = tester_with_time._setup_shot(
        [a_source], freq_hz=50.0, simulation_mode="steady-state"
    )
    time = tester_with_time.problem.time
    expected_wavelet = wavelet_helper("continuous_wave", time=time, freq_hz=50.0)
    expected_wavelet *= a_source.calculate_waveform_scale(dx=0.1)
    actual_wavelet = shot.wavelets.data[0]
    assert actual_wavelet.shape == expected_wavelet.shape
    np.testing.assert_allclose(shot.wavelets.data[0], expected_wavelet)


def test_setup_shot_wavelet_for_pulsed(tester_with_time, a_source):
    """_setup_shot should create a wavelet for the pulsed shot.

    This is tested here for a single source. Multiple sources are tested in
    test_shot.py.
    """
    shot = tester_with_time._setup_shot(
        [a_source], freq_hz=50.0, simulation_mode="pulsed"
    )
    time = tester_with_time.problem.time
    expected_wavelet = wavelet_helper("tone_burst", time=time, freq_hz=50.0)
    expected_wavelet *= a_source.calculate_waveform_scale(dx=0.1)
    actual_wavelet = shot.wavelets.data[0]
    assert actual_wavelet.shape == expected_wavelet.shape
    np.testing.assert_allclose(shot.wavelets.data[0], expected_wavelet)


@pytest.mark.parametrize("sim_mode", ["steady-state", "pulsed"])
def test_setup_sub_problem(tester_with_time, sim_mode):
    """Verify that the method executes the expected steps.

    The method should: 1) ensure a source, 2) set up a shot, and 3) returns a new
    sub-problem.
    """
    problem = tester_with_time.problem
    assert problem.acquisitions.num_shots == 0
    sub_problem = tester_with_time._setup_sub_problem(simulation_mode=sim_mode)
    assert len(tester_with_time.sources) == 1
    assert problem.acquisitions.num_shots == 1
    assert sub_problem.shot_id == problem.acquisitions.shot_ids[0]


def test_create_pde(tester_with_time):
    """_create_pde should create a new IsoAcousticDevito operator."""
    grid = tester_with_time.problem.grid
    pde = tester_with_time._create_pde()
    assert isinstance(pde, stride.IsoAcousticDevito)
    assert pde.space == grid.space
    assert pde.time == grid.time


@pytest.mark.parametrize("sim_mode", ["steady-state", "pulsed"])
def test_execute_pde(tester_with_time, fake_pde, sim_mode):
    """Verify the parameters of the call to the pde."""
    sub_problem = tester_with_time._setup_sub_problem(simulation_mode=sim_mode)
    test_save_bounds = (1, 100)
    test_undersampling = 7
    test_wavefield_slice = (slice(2, 1000),)
    n_jobs = 237
    result = tester_with_time._execute_pde(
        pde=fake_pde,
        sub_problem=sub_problem,
        save_bounds=test_save_bounds,
        save_undersampling=test_undersampling,
        wavefield_slice=test_wavefield_slice,
        n_jobs=n_jobs,
    )
    assert result == fake_pde.return_value
    kwargs = fake_pde.last_kwargs
    problem = tester_with_time.problem
    assert kwargs["problem"] == sub_problem
    assert kwargs["vp"] == problem.medium.fields["vp"]
    assert kwargs["rho"] == problem.medium.fields["rho"]
    assert kwargs["alpha"] == problem.medium.fields["alpha"]
    assert kwargs["boundary_type"] == "complex_frequency_shift_PML_2"
    assert kwargs["diff_source"]
    assert kwargs["save_wavefield"]
    assert kwargs["save_bounds"] == test_save_bounds
    assert kwargs["save_undersampling"] == test_undersampling
    assert kwargs["wavefield_slice"] == test_wavefield_slice
    assert kwargs["devito_args"]["nthreads"] == n_jobs


def test_wavefield_slice(base_tester):
    """Verify that _wavefield_slice returns the expected slices."""
    expected_slices = (
        slice(0, None),  # time
        slice(60, 81),  # X
        slice(60, 91),  # Y
        slice(60, 101),  # Z
    )
    actual_slices = base_tester._wavefield_slice()
    assert actual_slices == expected_slices


@pytest.mark.parametrize("selected_axis", [0, 1, 2])
def test_wavefield_slice_selects_correct_axis(
    tester_with_fixed_extra_space, selected_axis
):
    """Verify that _wavefield_slice modifies the right axis."""
    updated_slices = tester_with_fixed_extra_space._wavefield_slice(
        slice_axis=selected_axis,
        slice_position=tester_with_fixed_extra_space.origin[selected_axis],
    )
    assert updated_slices[selected_axis + 1] == slice(10, 11, None)


@pytest.mark.parametrize("delta_position", [1, 3, 9])
def test_wavefield_slice_selects_right_position(
    delta_position, tester_with_fixed_extra_space
):
    """Verify that _wavefield_slice modifies the right position."""
    dx = tester_with_fixed_extra_space.dx
    origin = tester_with_fixed_extra_space.origin
    print(dx, origin)
    selected_axis = 1
    updated_slices = tester_with_fixed_extra_space._wavefield_slice(
        slice_axis=selected_axis,
        slice_position=delta_position * dx + origin[selected_axis],
    )
    expected_updated_slice = slice(10 + delta_position, 11 + delta_position, None)
    assert updated_slices[selected_axis + 1] == expected_updated_slice


def test_get_steady_state_recording_time_bounds(tester_with_time):
    """Verify that expected time bounds for steady-state are returned."""
    time = tester_with_time.problem.time
    expected_bounds = (time.num - 24, time.num - 1)
    actual_bounds = tester_with_time._get_steady_state_recording_time_bounds(
        ppp=8, n_cycles=3
    )
    assert actual_bounds == expected_bounds


def test_get_pulsed_recording_time_bounds(tester_with_time):
    """Verify that expected time bounds for pulsed are returned."""
    time = tester_with_time.problem.time
    expected_bounds = (0, time.num - 1)
    actual_bounds = tester_with_time._get_pulsed_recording_time_bounds()
    assert actual_bounds == expected_bounds


def test_simulate_steady_state_pde_args(base_tester, fake_pde, monkeypatch):
    """Verify the pde call args from simulate_steady_state."""
    monkeypatch.setattr(base_tester, "_create_pde", lambda: fake_pde)
    test_ppp = 9
    test_n_cycles = 4
    test_undersampling = 7
    _ = base_tester.simulate_steady_state(
        points_per_period=test_ppp,
        n_cycles_steady_state=test_n_cycles,
        recording_time_undersampling=test_undersampling,
    )

    pde_kwargs = fake_pde.last_kwargs
    expected_bounds = base_tester._get_steady_state_recording_time_bounds(
        test_ppp, test_n_cycles
    )
    assert pde_kwargs["save_bounds"] == expected_bounds
    expected_wavefield = base_tester._wavefield_slice()
    assert pde_kwargs["wavefield_slice"] == expected_wavefield
    assert pde_kwargs["save_undersampling"] == test_undersampling


def test_simulate_steady_state_result(base_tester, fake_pde, monkeypatch):
    """Verify the elements of the simulate_steady_state result."""
    monkeypatch.setattr(base_tester, "_create_pde", lambda: fake_pde)
    result = base_tester.simulate_steady_state(
        points_per_period=9, n_cycles_steady_state=4, recording_time_undersampling=7
    )

    assert isinstance(result, SteadyStateResult)
    assert result.scenario == base_tester
    assert result.center_frequency == 5e5
    assert result.pde == fake_pde
    problem = result.scenario.problem
    assert result.shot.id == problem.acquisitions.shot_ids[0]
    assert result.traces == fake_pde.return_value


def test_simulate_steady_state_result_wavefield(base_tester, fake_pde, monkeypatch):
    """Verify the result wavefield has been reshaped as expected."""
    monkeypatch.setattr(base_tester, "_create_pde", lambda: fake_pde)
    result = base_tester.simulate_steady_state(
        points_per_period=9, n_cycles_steady_state=4, recording_time_undersampling=7
    )

    # drop the final timestep, then swap axes
    expected_wavefield = np.expand_dims(np.arange(8).reshape((2, 4)).T, 1)
    np.testing.assert_array_equal(result.wavefield, expected_wavefield)


def test_simulate_pulse_pde_args(base_tester, fake_pde, monkeypatch):
    """Verify the pde call args from simulate_pulse"""
    monkeypatch.setattr(base_tester, "_create_pde", lambda: fake_pde)
    test_ppp = 9
    test_sim_time = 3e-4
    test_undersampling = 7
    _ = base_tester.simulate_pulse(
        points_per_period=test_ppp,
        simulation_time=test_sim_time,
        recording_time_undersampling=test_undersampling,
    )

    pde_kwargs = fake_pde.last_kwargs

    expected_bounds = base_tester._get_pulsed_recording_time_bounds()
    assert pde_kwargs["save_bounds"] == expected_bounds
    expected_wavefield = base_tester._wavefield_slice()
    assert pde_kwargs["wavefield_slice"] == expected_wavefield
    assert pde_kwargs["save_undersampling"] == test_undersampling


def test_simulate_pulse_result(base_tester, fake_pde, monkeypatch):
    """Verify the elements of the simulate_pulse result"""
    monkeypatch.setattr(base_tester, "_create_pde", lambda: fake_pde)
    result = base_tester.simulate_pulse(
        points_per_period=9, simulation_time=3e-4, recording_time_undersampling=7
    )

    assert isinstance(result, PulsedResult)
    assert result.scenario == base_tester
    assert result.center_frequency == 5e5
    assert result.pde == fake_pde
    problem = result.scenario.problem
    assert result.shot.id == problem.acquisitions.shot_ids[0]
    assert result.traces == fake_pde.return_value


def test_simulate_pulse_result_wavefield(base_tester, fake_pde, monkeypatch):
    """Verify the result wavefield has been reshaped as expected"""
    monkeypatch.setattr(base_tester, "_create_pde", lambda: fake_pde)
    result = base_tester.simulate_pulse(
        points_per_period=9, simulation_time=3e-4, recording_time_undersampling=7
    )

    # drop the final timestep, then swap axes
    expected_wavefield = np.expand_dims(np.arange(8).reshape((2, 4)).T, 1)
    np.testing.assert_array_equal(result.wavefield, expected_wavefield)


def test_get_field_data(base_tester):
    """Verify that get_field_data returns the correct field data."""
    test_field = 4.2 * np.ones((21, 31, 41))
    base_tester.problem.medium.fields["vp"].data[:] = test_field
    field_value = base_tester.get_field_data("vp")
    np.testing.assert_allclose(field_value, test_field)
    assert isinstance(field_value, np.ndarray)


@pytest.mark.parametrize(
    "slice_axis, slice_position",
    [(3, 0.0), (1.5, 0.0), (0, 2.5), (1, -0.5), (2, 4.5), (None, 0.05), (1, None)],
)
def test_validate_slice_args_raises_error(base_tester, slice_axis, slice_position):
    """Verify that invalid slice arguments raise errors."""
    with pytest.raises(ValueError):
        base_tester._validate_slice_args(slice_axis, slice_position)
