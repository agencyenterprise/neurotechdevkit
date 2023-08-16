import numpy as np
import pytest
import stride

from neurotechdevkit.grid import Grid
from neurotechdevkit.problem import Problem
from neurotechdevkit.scenarios import _shots
from neurotechdevkit.scenarios._shots import (
    _add_point_transducers_to_geometry,
    _add_sources_to_geometry,
    _build_shot_wavelets_array,
    _create_delayed_source_wavelets,
    _get_wavelets_for_source,
    create_shot,
)


class FakeSource:
    """A fake source class that can be used for tests.

    This class includes the minimal set of properties and functions required
    for existing tests.
    """

    def __init__(self, scaling):
        self._scaling = scaling

    @property
    def point_source_delays(self):
        return np.array([1, 3, 5])

    @property
    def num_points(self):
        return 3

    @property
    def coordinates(self):
        return np.array([[0.0, 0.1], [0.5, 0.3], [1.1, 0.2]])

    def calculate_waveform_scale(self, dx):
        return self._scaling / dx


@pytest.fixture
def fake_source_1():
    """A fake source object that can be used for tests."""
    return FakeSource(1.618)


@pytest.fixture
def fake_source_2():
    """A fake source object that can be used for tests."""
    return FakeSource(2.5)


@pytest.fixture
def a_problem():
    """A minimal problem which can be used for tests."""
    grid = Grid.make_grid(
        extent=np.array([1.5, 0.6]),
        speed_water=100000,
        ppw=2,
        center_frequency=5e5,
    )
    grid.time = stride.Time(0.0, step=1e-3, num=100)
    problem = Problem(grid=grid)
    return problem


def fake_source_wavelet_array(dt, n):
    """Compute the unscaled wavelet array that matches a FakeSource."""
    delays = (np.array([1, 3, 5]) / dt).astype(int)
    return np.array([[0] * delays[i] + list(range(n - delays[i])) for i in range(3)])


def test_create_shot_adds_to_geometry(a_problem, fake_source_1, fake_source_2):
    """The shot should include source locations which match the problem geometry."""
    sources = [fake_source_1, fake_source_2]
    receiver_coords = [np.zeros(shape=2)]
    origin = np.array([-0.2, -0.1])
    wavelet = np.arange(100)
    shot = create_shot(a_problem, sources, receiver_coords, origin, wavelet, dx=0.1)
    assert len(a_problem.geometry.locations) == (
        fake_source_1.num_points + fake_source_2.num_points + len(receiver_coords)
    )
    assert (shot.sources + shot.receivers) == a_problem.geometry.locations


def test_create_shot_creates_wavelet(a_problem, fake_source_1, fake_source_2):
    """The shot should setup the wavelet array."""
    sources = [fake_source_1, fake_source_2]
    origin = np.array([-0.2, -0.1])
    wavelet = np.arange(100)
    shot = create_shot(a_problem, sources, [], origin, wavelet, dx=0.1)
    assert shot.wavelets.data.shape == (6, 100)
    assert np.any(shot.wavelets.data.shape != 0.0)


def test_create_shot_has_correct_shot_properties(a_problem, fake_source_1):
    """The shot should have the correct attributes assigned."""
    wavelet = np.arange(100)
    shot = create_shot(
        a_problem, [fake_source_1], [], np.array([0.0, 0.0]), wavelet, dx=0.1
    )
    assert shot.id == 0
    assert shot.problem == a_problem
    assert shot.geometry == a_problem.geometry
    assert len(shot.receivers) == 0


def test_create_shot_assigns_acquisition(a_problem, fake_source_1):
    """The shot should be added to the problem acquisitions."""
    wavelet = np.arange(100)
    shot = create_shot(
        a_problem, [fake_source_1], [], np.array([0.0, 0.0]), wavelet, dx=0.1
    )
    assert a_problem.acquisitions.get(0) == shot


def test_add_sources_to_geometry_returns_all_locations(
    a_problem, fake_source_1, fake_source_2
):
    """Verify that locations are added for all sources and returned."""
    origin = np.array([0.0, 0.0])
    point_transducers = _add_sources_to_geometry(
        a_problem, [fake_source_1, fake_source_2], origin
    )
    assert len(point_transducers) == 6
    assert a_problem.geometry.locations == point_transducers


def test_add_sources_to_geometry_shifts_origin(a_problem, fake_source_1):
    """Verify that locations are added for all sources and returned."""
    origin = np.array([-0.3, -0.1])
    point_transducers = _add_sources_to_geometry(a_problem, [fake_source_1], origin)
    first_coords = point_transducers[0].coordinates
    expected = fake_source_1.coordinates[0] - origin
    np.testing.assert_equal(first_coords, expected)


def test_add_point_transducers_to_geometry(a_problem):
    """The point source coords should be added to the problem geometry and returned."""
    coords = np.array([[0.0, 0.1], [0.5, 0.3], [1.1, 0.2]])
    point_transducers = _add_point_transducers_to_geometry(a_problem, coords)
    assert len(point_transducers) == 3
    assert [p.id for p in point_transducers] == [0, 1, 2]
    assert a_problem.geometry.locations == point_transducers
    for n, transducer in enumerate(point_transducers):
        np.testing.assert_equal(transducer.coordinates, coords[n])


def test_add_point_transducers_to_geometry_with_preexisting_locations(a_problem):
    """Existing locations should be preserved without problem.

    New location ids should continue beyond the last index of the existing locations.
    """
    coords = np.ones((3, 2))
    a_problem.transducers.default()
    a_problem.geometry.add(0, a_problem.transducers.get(0), np.array([0.0, 0.0]))
    a_problem.geometry.add(1, a_problem.transducers.get(0), np.array([0.1, 0.1]))
    point_transducers = _add_point_transducers_to_geometry(a_problem, coords)
    assert len(point_transducers) == 3
    assert [p.id for p in point_transducers] == [2, 3, 4]
    assert a_problem.geometry.locations[2:] == point_transducers


def test_build_shot_wavelets_array_concatenates_source_wavelets(monkeypatch):
    """Verify that the arrays for each source are concatenated together and returned.

    The function should build a wavelet array for each source and concatenate them
    together.
    """
    fake_wavelets = [
        1 * np.ones((2, 10)),
        2 * np.ones((3, 10)),
        3 * np.ones((7, 10)),
    ]
    wavelet_iter = iter(fake_wavelets)
    monkeypatch.setattr(
        _shots, "_get_wavelets_for_source", lambda *a, **kw: next(wavelet_iter)
    )
    fake_sources = ["s1", "s2", "s3"]
    wavelets = _build_shot_wavelets_array(np.ones((10,)), fake_sources, 1.0, 1.0)
    expected = np.concatenate(fake_wavelets)
    np.testing.assert_allclose(wavelets, expected)


def test_build_shot_wavelets_array_passes_args_for_source(fake_source_1, fake_source_2):
    """Verify that the parameters for sources are correctly passed to child fns.

    The verification is done by testing that the wavelet arrays for two sources are
    correct and that the wavelets are concatenated in the expected order.
    """
    fake_wavelet = np.arange(12)
    dt = 0.5
    dx = 1.0
    wavelets = _build_shot_wavelets_array(
        fake_wavelet, [fake_source_1, fake_source_2], dx, dt
    )
    expected = np.concatenate(
        [
            1.618 / dx * fake_source_wavelet_array(dt, 12),
            2.5 / dx * fake_source_wavelet_array(dt, 12),
        ]
    )
    np.testing.assert_allclose(wavelets, expected)


def test_get_wavelets_for_source(fake_source_1):
    """Verify that the wavelets are delayed and scaled properly for a single source."""
    fake_wavelet = np.arange(12)
    dt = 0.5
    dx = 2.0
    wavelets = _get_wavelets_for_source(fake_wavelet, fake_source_1, dx, dt)
    expected = 1.618 / dx * fake_source_wavelet_array(dt, 12)
    np.testing.assert_allclose(wavelets, expected)


def test_create_delayed_source_wavelets_with_zero_delays():
    """Verify that the wavelet is not delayed when all delays are zero."""
    dummy_wavelet = np.ones(shape=(4))
    fake_delays = np.zeros(shape=(5))
    fake_dt = 1.0
    delayed_wavelets = _create_delayed_source_wavelets(
        dummy_wavelet, fake_delays, fake_dt
    )
    np.testing.assert_allclose(delayed_wavelets, np.ones(shape=(5, 4)))


def test_create_delayed_source_wavelets_with_delays():
    """Verify that wavelets are shifted correctly when delays are nonzero."""
    dummy_wavelet = np.arange(10, 14)
    expected_array = np.array(
        [[10, 11, 12, 13], [0, 10, 11, 12], [0, 0, 10, 11], [0, 0, 0, 10]]
    )
    fake_delays = np.arange(0, 4)
    fake_dt = 1.0
    delayed_wavelets = _create_delayed_source_wavelets(
        dummy_wavelet, fake_delays, fake_dt
    )
    np.testing.assert_allclose(delayed_wavelets, expected_array)
