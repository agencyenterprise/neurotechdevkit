from __future__ import annotations

import warnings

import numpy as np
import pytest

from neurotechdevkit.sources import (
    FocusedSource2D,
    FocusedSource3D,
    PhasedArraySource2D,
    PhasedArraySource3D,
    PlanarSource2D,
    PlanarSource3D,
    PointSource2D,
    PointSource3D,
    _rotate_2d,
    _rotate_3d,
)


class TestRotMat2D:
    @pytest.mark.parametrize(
        "coords, theta, expected",
        [
            (np.array([1, 0]), 0, np.array([1, 0])),
            (np.array([1, 0]), np.pi / 2, np.array([0, 1])),
            (np.array([0, 1]), np.pi / 2, np.array([-1, 0])),
            (np.array([1, 1]), np.pi / 4, np.array([0, np.sqrt(2)])),
            (np.array([1, 1]), np.pi / 2, np.array([-1, 1])),
            (np.array([[1, 0], [0, 1]]), np.pi / 2, np.array([[0, 1], [-1, 0]])),
        ],
    )
    def test_rotate_2d(self, coords, theta, expected):
        np.testing.assert_allclose(_rotate_2d(coords, theta), expected, atol=1e-10)


class TestRotMat3D:
    def test_rot_matrix_valid_input(self):
        # Test with a valid input
        axis = np.array([1, 0, 0])
        theta = (np.pi / 2,)
        expected_result = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        coords = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        np.testing.assert_allclose(_rotate_3d(coords, axis, theta), expected_result)

    def test_rot_matrix_invalid_input_axis_size(self):
        # Test with invalid input where axis has size not equal to 3
        axis = np.array([1, 0])
        theta = np.pi / 2
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            _rotate_3d(coords=coords, axis=axis, theta=theta)

    def test_rot_matrix_invalid_input_axis_norm_zero(self):
        # Test with invalid input where axis has norm equal to zero
        axis = np.array([0, 0, 0])
        theta = np.pi / 2
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            _rotate_3d(coords=coords, axis=axis, theta=theta)


class Source2DTestMixin:
    def test_position_property(self):
        """Verify that .position matches the value received in the constructor."""
        source = self.create_test_source(position=np.array([-2.0, 3.0]))
        np.testing.assert_allclose(source.position, np.array([-2.0, 3.0]))

    def test_unit_direction_is_normalized(self):
        """Verify that .unit_direction is normalize."""
        source = self.create_test_source(direction=np.array([2.0, 3.0]))
        np.testing.assert_allclose(np.linalg.norm(source.unit_direction), 1.0)

    def test_unit_direction_points_in_the_expected_direction(self):
        """Verify that .unit_direction is parallel to the given parameter."""
        test_direction = np.array([1.0, 1.0])
        source = self.create_test_source(direction=test_direction)
        np.testing.assert_allclose(0.0, np.cross(source.unit_direction, test_direction))
        assert np.dot(source.unit_direction, test_direction) > 0.0

    def test_aperture_property(self):
        """Verify that .aperture matches the value received in the constructor."""
        source = self.create_test_source(aperture=7.0)
        assert source.aperture == 7.0

    def test_num_points_property(self):
        """Verify that .num_points matches the value received in the constructor."""
        source = self.create_test_source(num_points=1234)
        assert source.num_points == 1234

    def test_coordinates_has_correct_shape(self):
        """Verify that .coordinates has the expected shape."""
        warnings.simplefilter("ignore")
        source = self.create_test_source(num_points=1234)
        assert source.coordinates.shape == (1234, 2)

    def test_point_source_delays_property(self):
        """Verify that .point_source_delays are set up correctly in the constructor."""
        source = self.create_test_source(num_points=10)
        np.testing.assert_allclose(source.point_source_delays, np.zeros(shape=10))

    def test_point_source_delays_property_positive_delay(self):
        """Verify that .point_source_delays are set up correctly in the constructor."""
        source = self.create_test_source(num_points=10, delay=123)
        np.testing.assert_allclose(
            source.point_source_delays, np.full(shape=10, fill_value=123)
        )

    def test_point_source_delays_property_has_correct_shape(self):
        """Verify that .point_source_delays have the correct shape."""
        source = self.create_test_source(num_points=3)
        assert source.point_source_delays.shape[0] == 3

    def test_delay_property(self):
        """Verify that the .delay property is set up correctly."""
        source = self.create_test_source(delay=123)
        assert source.delay == 123

    def test_negative_delay_raises_error(self):
        """Verify that a negative delay raises an error."""
        with pytest.raises(ValueError):
            self.create_test_source(delay=-1)


class TestFocusedSource2D(Source2DTestMixin):
    @staticmethod
    def create_test_source(
        *,
        position: np.ndarray = np.array([1.0, 2.0]),
        direction: np.ndarray = np.array([0.0, 1.0]),
        aperture: float = 15.0,
        focal_length: float = 10.0,
        num_points: int = 500,
        delay: float = 0.0,
    ) -> FocusedSource2D:
        """Creates a FocusedSource2D source with default parameters.

        This utility function gives default values for each parameter
        and allows us to choose to only specify one or a few parameters
        at a time.

        See `neurotechdevkit.sources.FocusedSource2D` for more details.

        Args:
            position: a numpy float array of shape (2,) indicating the
                coordinates of the point at the bottom/center of the
                arc source.
            direction: a numpy float array of shape (2,) indicating the
                orientation of the source.
            aperture: the width of the source.
            focal_length: the distance from `position` ot the focal point.
            num_points: the number of point sources to use when simulating the source.
            delay: the delay (in seconds) for the whole source.

        Returns:
            A FocusedSource2D instance with the requested parameters.
        """
        return FocusedSource2D(
            position=position,
            direction=direction,
            aperture=aperture,
            focal_length=focal_length,
            num_points=num_points,
            delay=delay,
        )

    @staticmethod
    @pytest.fixture(scope="module")
    def dense_source():
        """Creates a FocusedSource2D with a large number of source points.

        A dense set of source points means that the points will extend very near
        to the edges and the center of of the arc. This is useful for tests that verify
        dimensions of the source point cloud.

        Returns:
            A FocusedSource2D instance with a dense source point cloud.
        """
        return TestFocusedSource2D.create_test_source(
            position=np.array([0.0, 1.0]),
            direction=np.array([0.0, 1.0]),
            aperture=5.0,
            focal_length=4.0,
            num_points=2_000,
        )

    def test_focal_length_property(self):
        """Verify that .focal_length matches the value received in the constructor."""
        source = self.create_test_source(focal_length=17.0)
        assert source.focal_length == 17.0

    def test_coordinates_with_aperture_too_large(self):
        """Should raise a ValueError if aperture is larger than twice focal_length."""
        with pytest.raises(ValueError):
            self.create_test_source(aperture=10.0, focal_length=4.0)

    def test_coordinates_position(self, dense_source):
        """Verify that the coordinates agree with the specified position.

        Since the direction of dense_source is along the y-axis, we can
        expect that the center should be very close to the point which
        has the minimum y-value.
        """
        coords = dense_source.coordinates
        center_arg = np.argmin(coords[:, 1])
        center_coords = coords[center_arg]
        np.testing.assert_allclose(center_coords, dense_source.position, atol=1e-2)

    def test_coordinates_direction(self, dense_source):
        """Verify that the coordinates agree with the specified direction.

        Assuming that the direction of dense_source is along the y-axis, we
        can expect that the mean coordinate over the x-direction should
        agree with the specified position.
        """
        coords = dense_source.coordinates
        x_center = np.mean(coords[:, 0])
        np.testing.assert_allclose(x_center, dense_source.position[0], atol=1e-7)

    def test_coordinates_aperture(self, dense_source):
        """Verify that the actual aperture matches the specified aperture.

        Since the direction of dense_source is along the y-axis, we can expect
        that the range of x coordinates should be very close to the specified aperture.
        """
        coords = dense_source.coordinates
        x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
        np.testing.assert_allclose(x_range, dense_source.aperture)

    def test_coordinates_focal_length(self, dense_source):
        """Verify that the coordinates agree with the specified focal_length.

        If d is the depth of the source, r is the radius, and a is the aperture,
        then from the Pythagorean theorem, we have:
        $$
        r^2 = (a/2)^2 + (r-d)^2
        $$
        And solving for r, we get:
        $$
        r = ((a/2)^2 + h^2) / (2*h)
        $$
        """
        coords = dense_source.coordinates
        a = dense_source.aperture
        h = np.max(coords[:, 1]) - np.min(coords[:, 1])
        r = ((a / 2) ** 2 + h**2) / (2 * h)
        np.testing.assert_allclose(r, dense_source.focal_length, rtol=1e-6)

    def test_calculate_waveform_scale_for_semicircle(self):
        """Verify that calculate_waveform_scale returns the expected scale given a
        semicircular source.

        The length of a semicircle is pi*r.
        """
        radius = 10.0
        num_points = 800
        dx = 0.01
        source = self.create_test_source(
            aperture=2 * radius, focal_length=radius, num_points=num_points
        )
        scale = source.calculate_waveform_scale(dx=dx)
        grid_density = 1 / dx
        point_density = num_points / (np.pi * radius)
        np.testing.assert_allclose(scale, grid_density / point_density)

    def test_calculate_waveform_scale_for_60_deg(self):
        """Verify that calculate_waveform_scale returns the expected scale given a
        an angle to the arc edge of 60 degrees.

        sin(60°) is 1/2, so the arc height is radius/2, and the length for this
        bowl is 2*pi/3*r.
        """
        radius = 10.0
        num_points = 800
        dx = 0.01
        source = self.create_test_source(
            aperture=np.sqrt(3) * radius,
            focal_length=radius,
            num_points=num_points,
        )
        scale = source.calculate_waveform_scale(dx=dx)
        grid_density = 1 / dx
        point_density = num_points / (2 * np.pi / 3 * radius)
        np.testing.assert_allclose(scale, grid_density / point_density)


class Source3DTestMixin:
    def test_position_property(self):
        """Verify that .position matches the value received in the constructor."""
        source = self.create_test_source(position=np.array([-2.0, 3.0, 7.0]))
        np.testing.assert_allclose(source.position, np.array([-2.0, 3.0, 7.0]))

    def test_unit_direction_is_normalized(self):
        """Verify that .unit_direction is normalize."""
        source = self.create_test_source(direction=np.array([2.0, 3.0, 1.0]))
        np.testing.assert_allclose(np.linalg.norm(source.unit_direction), 1.0)

    def test_unit_direction_points_in_the_expected_direction(self):
        """Verify that .unit_direction is parallel to the given parameter."""
        test_direction = np.array([1.0, 1.0, 0.0])
        source = self.create_test_source(direction=test_direction)
        np.testing.assert_allclose(0.0, np.cross(source.unit_direction, test_direction))
        assert np.dot(source.unit_direction, test_direction) > 0.0

    def test_aperture_property(self):
        """Verify that .aperture matches the value received in the constructor."""
        source = self.create_test_source(aperture=7.0)
        assert source.aperture == 7.0

    def test_num_points_property(self):
        """Verify that .num_points matches the value received in the constructor."""
        source = self.create_test_source(num_points=1234)
        assert source.num_points == 1234

    def test_coordinates_has_correct_shape(self):
        """Verify that coordinates has the expected shape."""
        source = self.create_test_source(num_points=1234)
        assert source.coordinates.shape == (1234, 3)

    def test_point_source_delays_property(self):
        """Verify that .point_source_delays are set up correctly in the constructor."""
        source = self.create_test_source(num_points=10)
        np.testing.assert_allclose(source.point_source_delays, np.zeros(shape=10))

    def test_delay_property(self):
        """Verify that the .delay property is set up correctly."""
        source = self.create_test_source(delay=123)
        assert source.delay == 123


class TestFocusedSource3D(Source3DTestMixin):
    @staticmethod
    def create_test_source(
        *,
        position: np.ndarray = np.array([1.0, 2.0, 3.0]),
        direction: np.ndarray = np.array([1.0, 1.0, 0]),
        aperture: float = 15.0,
        focal_length: float = 10.0,
        num_points: int = 1000,
        delay: float = 0.0,
    ) -> FocusedSource3D:
        """Creates a FocusedSource3D source with default parameters.

        This utility function gives default values for each parameter
        and allows us to choose to only specify one or a few parameters
        at a time.

        See `neurotechdevkit.sources.FocusedSource3D` for more details.

        Args:
            position: a numpy float array of shape (3,) indicating the
                coordinates of the point at the bottom/center of the
                bowl source.
            direction: a numpy float array of shape (3,) indicating the
                orientation of the source.
            aperture: the width of the source.
            focal_length: the distance from `position` ot the focal point.
            num_points: the number of point sources to use when simulating the source.
            delay: the delay (in seconds) for the whole source.

        Returns:
            A FocusedSource3D instance with the requested parameters.
        """
        return FocusedSource3D(
            position=position,
            direction=direction,
            aperture=aperture,
            focal_length=focal_length,
            num_points=num_points,
            delay=delay,
        )

    @staticmethod
    @pytest.fixture(scope="module")
    def dense_source():
        """Creates a FocusedSource3D with a large number of source points.

        A dense set of source points means that the points will extend very near
        to the edge of the spherical size as well as to the center of the slice.
        This is useful for tests that verify dimensions of the source point cloud.

        This fixture has a module scope because it takes a couple of seconds to build
        the dense source point cloud.

        Returns:
            A FocusedSource3D instance with a dense source point cloud.
        """
        return TestFocusedSource3D.create_test_source(
            position=np.array([0.0, 1.0, 2.0]),
            direction=np.array([0.0, 1.0, 0.0]),
            aperture=5.0,
            focal_length=4.0,
            num_points=20_000,
        )

    def test_focal_length_property(self):
        """Verify that .focal_length matches the value received in the constructor."""
        source = self.create_test_source(focal_length=17.0)
        assert source.focal_length == 17.0

    def test_coordinates_with_aperture_too_large(self):
        """Should raise a ValueError if aperture is larger than twice focal_length."""
        with pytest.raises(ValueError):
            self.create_test_source(aperture=10.0, focal_length=4.0)

    def test_coordinates_aperture(self, dense_source):
        """Verify that the actual aperture matches the specified aperture.

        Since the direction of dense_source is along the y-axis, we can expect
        that the range of x and z coordinates should be very close to the
        specified aperture.
        """
        coords = dense_source.coordinates
        x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
        z_range = np.max(coords[:, 2]) - np.min(coords[:, 2])
        np.testing.assert_allclose(x_range, dense_source.aperture, rtol=1e-2)
        np.testing.assert_allclose(z_range, dense_source.aperture, rtol=1e-2)

    def test_coordinates_position(self, dense_source):
        """Verify that the coordinates agree with the specified position.

        Since the direction of dense_source is along the y-axis, we can
        expect that the center should be very close to the point which
        has the minimum y-value.
        """
        coords = dense_source.coordinates
        center_arg = np.argmin(coords[:, 1])
        center_coords = coords[center_arg]
        np.testing.assert_allclose(center_coords, dense_source.position, rtol=1e-2)

    def test_coordinates_direction(self, dense_source):
        """Verify that the coordinates agree with the specified direction.

        Assuming that the direction of dense_source is along the y-axis, we
        can expect that the mean coordinate over the x- and z- directions should
        agree with the specified position.
        """
        coords = dense_source.coordinates
        x_center = np.mean(coords[:, 0])
        z_center = np.mean(coords[:, 2])
        np.testing.assert_allclose(x_center, dense_source.position[0], atol=1e-4)
        np.testing.assert_allclose(z_center, dense_source.position[2], rtol=1e-5)

    def test_coordinates_focal_length(self, dense_source):
        """Verify that the coordinates agree with the specified focal_length.

        If d is the depth of the source, r is the radius, and a is the aperture,
        then from the Pythagorean theorem, we have:
        $$
        r^2 = (a/2)^2 + (r-d)^2
        $$
        And solving for r, we get:
        $$
        r = ((a/2)^2 + h^2) / (2*h)
        $$
        """
        coords = dense_source.coordinates
        a = dense_source.aperture
        h = np.max(coords[:, 1]) - np.min(coords[:, 1])
        r = ((a / 2) ** 2 + h**2) / (2 * h)
        np.testing.assert_allclose(r, dense_source.focal_length, rtol=1e-4)

    def test_calculate_waveform_scale_for_hemisphere(self):
        """Verify that calculate_waveform_scale returns the expected scale given a
        hemispherical source.

        The surface area for a hemispherical bowl is 2*pi*r^2.
        """
        radius = 10.0
        num_points = 3000
        dx = 0.01
        source = self.create_test_source(
            aperture=2 * radius, focal_length=radius, num_points=num_points
        )
        scale = source.calculate_waveform_scale(dx=dx)
        grid_density = 1 / dx**2
        point_density = num_points / (2 * np.pi * radius**2)
        np.testing.assert_allclose(scale, grid_density / point_density)

    def test_calculate_waveform_scale_for_60_deg(self):
        """Verify that calculate_waveform_scale returns the expected scale given a
        an angle to the bowl edge of 60 degrees.

        sin(60°) is 1/2, so the bowl height is radius/2, and the surface area for this
        bowl is pi*r^2.
        """
        radius = 10.0
        num_points = 3000
        dx = 0.01
        source = self.create_test_source(
            aperture=np.sqrt(3) * radius,
            focal_length=radius,
            num_points=num_points,
        )
        scale = source.calculate_waveform_scale(dx=dx)
        grid_density = 1 / dx**2
        point_density = num_points / (np.pi * radius**2)
        np.testing.assert_allclose(scale, grid_density / point_density)

    def test_calculate_threshold(self):
        """Verify the result of the _calculate_threshold utility function."""
        threshold = FocusedSource3D._calculate_threshold(aperture=42.0, radius=42.0)
        np.testing.assert_allclose(threshold, 0.5 * (1 + np.sqrt(3 / 4)))
        threshold = FocusedSource3D._calculate_threshold(
            aperture=2 * np.sqrt(5.0), radius=3.0
        )
        np.testing.assert_allclose(threshold, 5 / 6)

    @pytest.mark.parametrize(
        "unit_direction,expected_axis,expected_theta",
        [
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]), np.pi / 2),
            (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.pi / 2),
            (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), np.pi),
            (np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.0, 0.0]), 0.0),
            (
                np.array([0.0, np.sqrt(2) / 2, np.sqrt(2) / 2]),
                np.array([1.0, 0.0, 0.0]),
                3 * np.pi / 4,
            ),
            (
                np.array([np.sqrt(2) / 2, 0.0, -np.sqrt(2) / 2]),
                np.array([0.0, -1.0, 0.0]),
                np.pi / 4,
            ),
        ],
    )
    def test_calculate_rotation_parameters_to(
        self, unit_direction, expected_axis, expected_theta
    ):
        """Verify that _rotation_parameters returns the expected parameters.

        The parameters should describe the rotation from -z to unit_direction.
        """
        axis, theta = FocusedSource3D._calculate_rotation_parameters(unit_direction)
        np.testing.assert_allclose(axis, expected_axis)
        np.testing.assert_allclose(theta, expected_theta)

    def test_calculate_rotation_parameters_raises_error_on_bad_input(self):
        """Verifies that a ValueError is raised if unit_direction is not normalized."""
        with pytest.raises(ValueError):
            FocusedSource3D._calculate_rotation_parameters(np.array([2.0, 2.0, 2.0]))


class TestPlanarSource2D(Source2DTestMixin):
    @staticmethod
    def create_test_source(
        *,
        position: np.ndarray = np.array([1.0, 2.0]),
        direction: np.ndarray = np.array([0.0, 1.0]),
        aperture: float = 15.0,
        num_points: int = 500,
        delay: float = 0.0,
    ) -> PlanarSource2D:
        """Creates a PlanarSource2D source with default parameters.

        This utility function gives default values for each parameter
        and allows us to choose to only specify one or a few parameters
        at a time.

        See `neurotechdevkit.sources.PlanarSource2D` for more details.

        Args:
            position: a numpy float array of shape (2,) indicating the
                coordinates of the point at the center of the line segment
                source.
            direction: a numpy float array of shape (2,) indicating the
                orientation of the source.
            aperture: the width of the source.
            num_points: the number of point sources to use when simulating the source.
            delay: the delay (in seconds) for the whole source.

        Returns:
            A PlanarSource2D instance with the requested parameters.
        """
        return PlanarSource2D(
            position=position,
            direction=direction,
            aperture=aperture,
            num_points=num_points,
            delay=delay,
        )

    @staticmethod
    @pytest.fixture(scope="module")
    def dense_source():
        """Creates a PlanarSource2D with a large number of source points.

        A dense set of source points means that the points will extend very near
        to the edge and center of the source. This is useful for tests that verify
        dimensions of the source point cloud.

        Returns:
            A PlanarSource2D instance with a dense source point cloud.
        """
        return TestPlanarSource2D.create_test_source(
            position=np.array([0.0, 1.0]),
            direction=np.array([1.0, 1.0]),
            aperture=5.0,
            num_points=2_000,
        )

    def test_coordinates_position(self, dense_source):
        """Verify that the coordinates agree with the specified position.

        The center should be the at mean of the points.
        """
        center_coords = np.mean(dense_source.coordinates, axis=0)
        np.testing.assert_allclose(center_coords, dense_source.position, atol=1e-7)

    def test_coordinates_direction(self, dense_source):
        """Verify that the coordinates agree with the specified direction.

        For the linear source, the direction should be perpendicular to the line.
        """
        coords = dense_source.coordinates
        in_line = coords[-1, :] - coords[0, :]
        np.testing.assert_allclose(
            0.0, np.dot(in_line, dense_source.unit_direction), atol=1e-7
        )

    def test_coordinates_aperture(self, dense_source):
        """Verify that the actual aperture matches the specified aperture.

        For the linear source, the aperture should match the distance between edges.
        """
        coords = dense_source.coordinates
        length = np.linalg.norm(coords[-1, :] - coords[0, :])
        np.testing.assert_allclose(length, dense_source.aperture)

    def test_calculate_waveform_scale(self):
        """Verify that calculate_waveform_scale returns the expected scale factor"""
        length = 42.0
        num_points = 800
        dx = 0.01
        source = self.create_test_source(aperture=length, num_points=num_points)
        scale = source.calculate_waveform_scale(dx=dx)
        grid_density = 1 / dx
        point_density = num_points / length
        np.testing.assert_allclose(scale, grid_density / point_density)


class TestPlanarSource3D(Source3DTestMixin):
    @staticmethod
    def create_test_source(
        *,
        position: np.ndarray = np.array([1.0, 2.0, 3.0]),
        direction: np.ndarray = np.array([1.0, 1.0, 0]),
        aperture: float = 15.0,
        focal_length: float = 10.0,
        num_points: int = 1000,
        delay: float = 0.0,
    ) -> PlanarSource3D:
        """Creates a PlanarSource3D source with default parameters.

        This utility function gives default values for each parameter
        and allows us to choose to only specify one or a few parameters
        at a time.

        See `neurotechdevkit.sources.PlanarSource3D` for more details.

        Args:
            position: a numpy float array of shape (3,) indicating the
                coordinates of the point at the bottom/center of the
                bowl source.
            direction: a numpy float array of shape (3,) indicating the
                orientation of the source.
            aperture: the width of the source.
            num_points: the number of point sources to use when simulating the source.
            delay: the delay (in seconds) for the whole source.

        Returns:
            A PlanarSource3D instance with the requested parameters.
        """
        return PlanarSource3D(
            position=position,
            direction=direction,
            aperture=aperture,
            num_points=num_points,
            delay=delay,
        )

    @staticmethod
    @pytest.fixture(scope="module")
    def dense_source():
        """Creates a PlanarSource3D with a large number of source points.

        A dense set of source points means that the points will extend very near
        to the edge and the center of the disk. This is useful for tests that verify
        dimensions of the source point cloud.

        This fixture has a module scope because it takes a couple of seconds to build
        the dense source point cloud.

        Returns:
            A PlanarSource3D instance with a dense source point cloud.
        """
        return TestPlanarSource3D.create_test_source(
            position=np.array([0.0, 1.0, 2.0]),
            direction=np.array([0.0, 1.0, 1.0]),
            aperture=5.0,
            focal_length=4.0,
            num_points=20_000,
        )

    def test_coordinates_position(self, dense_source):
        """Verify that the coordinates agree with the specified position.

        The center should be the at mean of the points.
        """
        center_coords = np.mean(dense_source.coordinates, axis=0)
        np.testing.assert_allclose(
            center_coords, dense_source.position, rtol=1e-4, atol=1e-4
        )

    @pytest.mark.parametrize(
        "idx1,idx2",
        [
            (0, -1),
            (50, 4_321),
            (17_654, 11_111),
            (5_000, 15_000),
        ],
    )
    def test_coordinates_direction(self, dense_source, idx1, idx2):
        """Verify that the coordinates agree with the specified direction.

        For the disk source, the direction should be perpendicular any points within
        the disk.
        """
        coords = dense_source.coordinates
        in_plane = coords[idx1, :] - coords[idx2, :]
        np.testing.assert_allclose(
            0.0, np.dot(in_plane, dense_source.unit_direction), atol=1e-7
        )

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_coordinates_aperture(self, dense_source, axis):
        """Verify that the actual aperture matches the specified aperture.

        For the disk source, the aperture should match the distance between opposite
        edges. These edges can be obtained by the extremes of the disk along any axis.
        """
        coords = dense_source.coordinates
        min_idx = np.argmin(coords[:, axis])
        max_idx = np.argmax(coords[:, axis])
        length = np.linalg.norm(coords[max_idx, :] - coords[min_idx, :])
        np.testing.assert_allclose(length, dense_source.aperture, rtol=1e-4)

    def test_calculate_waveform_scale(self):
        """Verify that calculate_waveform_scale returns the expected scale factor.

        The surface area for a disk is pi*r^2.
        """
        radius = 21.0
        num_points = 800
        dx = 0.01
        source = self.create_test_source(aperture=radius * 2, num_points=num_points)
        scale = source.calculate_waveform_scale(dx=dx)
        grid_density = 1 / dx**2
        point_density = num_points / (np.pi * radius**2)
        np.testing.assert_allclose(scale, grid_density / point_density)


class TestPhasedArraySource2D(Source2DTestMixin):
    @staticmethod
    def create_test_source(
        *,
        position: np.ndarray = np.array([1.0, 2.0]),
        direction: np.ndarray = np.array([0.0, 1.0]),
        num_points: int = 1000,
        pitch: float = 0.01,
        num_elements: int = 5,
        tilt_angle: float = 60.0,
        element_width: float = 0.009,
        focal_length: float = np.inf,
        delay: float = 0.0,
        element_delays: np.ndarray | None = None,
    ) -> PhasedArraySource2D:
        """Creates a PhasedArraySource2D source with default parameters.

        This utility function gives default values for each parameter
        and allows us to choose to only specify one or a few parameters
        at a time.

        See `neurotechdevkit.sources.PhasedArraySource2D` for more details.

        Args:
            position: an array indicating the coordinates of the center of the source.
            direction: an array indicating the orientation of the source.
            num_points: the number of point sources to use when simulating the source.
            pitch: the distance between the center of neighboring elements.
            num_elements: the number of elements of the phased array.
            tilt_angle: the desired tilt angle of the wavefront.
            element_width: the width of each element of the array.
            focal_length: the focal length of the array.
            delay: the delay (in seconds) for the whole source.
            element_delays: the delay (in seconds) for each element of the phased array.

        Returns:
            A PhasedArraySource2D instance with the requested parameters.
        """
        return PhasedArraySource2D(
            position=position,
            direction=direction,
            num_points=num_points,
            pitch=pitch,
            num_elements=num_elements,
            tilt_angle=tilt_angle,
            element_width=element_width,
            focal_length=focal_length,
            delay=delay,
            element_delays=element_delays,
        )

    @staticmethod
    @pytest.fixture(scope="module")
    def dense_source():
        """Creates a PhasedArraySource2D with a large number of source points.

        A dense set of source points means that the points will extend very near
        to the edge of each element. This is useful for tests that verify
        dimensions of the source point line.

        This fixture has a module scope because it takes a couple of seconds to build
        the dense source point cloud.

        Returns:
            A PhasedArraySource2D instance with a dense source point cloud.
        """
        return TestPhasedArraySource2D.create_test_source(
            position=np.array([0.0, 1.0]),
            direction=np.array([1.0, 1.0]),
            num_points=2_000,
        )

    def test_num_points_property(self):
        """Verify that .num_points matches the value received in the constructor."""
        source = TestPhasedArraySource2D.create_test_source(
            num_points=10 * 13, num_elements=13
        )
        assert source.num_points == 130

    def test_point_source_delays_property_has_correct_shape(self):
        """Verify that .point_source_delays have the correct shape."""
        source = TestPhasedArraySource2D.create_test_source(
            num_points=3, num_elements=3
        )
        assert source.point_source_delays.shape[0] == 3

    def test_aperture_property(self):
        """Verify that .aperture is constructed correctly.
        Note that for PhasedArraySource2D aperture must be computed.
        """
        source = TestPhasedArraySource2D.create_test_source(
            pitch=1, num_elements=5, element_width=0.75
        )
        # spacing = pitch - element_width
        # aperture = pitch * num_elements - spacing
        assert source.aperture == 4.75

    def test_pitch_property(self):
        """Verify that .pitch matches the value received in the constructor."""
        source = TestPhasedArraySource2D.create_test_source(pitch=0.01)
        assert source.pitch == 0.01

    def test_num_elements_property(self):
        """Verify that .num_elements matches the value received in the constructor."""
        warnings.simplefilter("ignore")
        source = TestPhasedArraySource2D.create_test_source(num_elements=12)
        assert source.num_elements == 12

    def test_num_elements_raises_error(self):
        """Verify that value error is raised if num elements is one."""
        with pytest.raises(ValueError):
            TestPhasedArraySource2D.create_test_source(num_elements=1)

    def test_tilt_angle_property(self):
        """Verify that .tilt_angle matches the value received in the constructor."""
        source = TestPhasedArraySource2D.create_test_source(tilt_angle=12.34)
        assert source.tilt_angle == 12.34

    def test_spacing_property(self):
        """Verify that .spacing is set up correctly."""
        source = TestPhasedArraySource2D.create_test_source(
            pitch=0.1123, element_width=0.100
        )
        np.testing.assert_allclose(source.spacing, 0.0123)

    def test_focal_length_property(self):
        """Verify that .focal_length is set up correctly."""
        source = TestPhasedArraySource2D.create_test_source(focal_length=0.123)
        assert source.focal_length == 0.123

    def test_validate_num_points_warns_user(self):
        """Verify that the user is warned when the number of points is adjusted."""
        with pytest.warns(UserWarning):
            TestPhasedArraySource2D.create_test_source(num_points=13, num_elements=5)

    def test_validate_num_points_modify_value(self):
        """Verify that the number of points are truncated when needed."""
        warnings.simplefilter("ignore")
        source = TestPhasedArraySource2D.create_test_source(
            num_points=101, num_elements=10
        )
        assert source.num_points == 100

    def test_validate_num_points_does_not_modify_value(self):
        """Verify that the number of points is not modified if not required."""
        source = TestPhasedArraySource2D.create_test_source(
            num_points=100, num_elements=10
        )
        assert source.num_points == 100

    def test_validate_num_points_does_not_warns(self):
        """Verify that no warning is shown when no modification is required."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TestPhasedArraySource2D.create_test_source(num_points=100, num_elements=10)

    @pytest.mark.parametrize(
        "position, direction, focal_length, angle",
        [
            (np.array([0.0, 0.0]), np.array([0.0, 1.0]), np.sqrt(2), -45),
            (np.array([1 + 2 * np.sin(np.pi / 3), 0.0]), np.array([0.0, 1.0]), 2, 60),
            (np.array([0, 1 + 2 * np.cos(np.pi / 6)]), np.array([0.0, -1.0]), 2, 30),
            (np.array([2, 2]), np.array([-1, -1]), np.sqrt(2), 0),
        ],
    )
    def test_focal_point_is_computed_correctly(
        self, position, direction, focal_length, angle
    ):
        """Verify that the focal point is created correctly.
        Four test cases (all of them aim to the same focal point):

        Test 1: tilt angle affecting equally the x and y component.
        Test 2: displace in X axis to compensate for the tilt angle
        Test 3: displace in Y axis as a result from the negative unit direction.
        Test 4: Direction with x and y non-zero components.
        """
        source = TestPhasedArraySource2D.create_test_source(
            position=position,
            direction=direction,
            focal_length=focal_length,
            tilt_angle=angle,
        )
        np.testing.assert_allclose(source.focal_point, np.array([1, 1]))

    def test_focal_point_is_inf_for_unfocused_arrays(self):
        """Verify that focal point is (inf,inf) when unfocused."""
        source = TestPhasedArraySource2D.create_test_source(
            focal_length=np.inf,
        )
        assert all([np.isinf(x) for x in source.focal_point])

    @pytest.mark.parametrize(
        "tilt_angle, delay, expected_delays",
        [(0.0, 0.0, np.array([0.0, 0.0])), (30.0, 1.1, np.array([1.1, 2.1]))],
    )
    def test_point_source_delays_property(self, tilt_angle, delay, expected_delays):
        """Verify that the original .point_source_delays are set correctly."""
        source = TestPhasedArraySource2D.create_test_source(
            tilt_angle=tilt_angle,
            num_points=2,
            num_elements=2,
            pitch=1500.0 * 2.0,
            delay=delay,
        )
        np.testing.assert_allclose(
            source.point_source_delays, expected_delays, atol=1e-7
        )

    def test_point_source_delays_property_positive_delay(self):
        """Verify that .point_source_delays are set up correctly in the constructor."""
        source = TestPhasedArraySource2D.create_test_source(
            num_points=10, delay=1, tilt_angle=0
        )
        np.testing.assert_allclose(source.point_source_delays, np.ones(shape=10))

    def test_element_delays_property(self):
        """Verify that .element_delays property is set up correctly"""
        source = TestPhasedArraySource2D.create_test_source(
            num_points=10, num_elements=5, tilt_angle=0, element_delays=np.arange(0, 5)
        )
        np.testing.assert_array_equal(source.element_delays, np.array([0, 1, 2, 3, 4]))

    def test_coordinates_has_correct_shape(self):
        """Verify that .coordinates has the expected shape."""
        source = TestPhasedArraySource2D.create_test_source(
            num_points=1200, num_elements=10
        )
        assert source.coordinates.shape == (1200, 2)

    def test_coordinates_position(self, dense_source):
        """Verify that the coordinates agree with the specified position.

        The center should be the at mean of the points.
        """
        center_coords = np.mean(dense_source.coordinates, axis=0)
        np.testing.assert_allclose(center_coords, dense_source.position, atol=1e-7)

    def test_coordinates_direction(self, dense_source):
        """Verify that the coordinates agree with the specified direction.

        For the phased array source, the direction should be perpendicular to the line.
        """
        coords = dense_source.coordinates
        in_line = coords[-1, :] - coords[0, :]
        np.testing.assert_allclose(
            0.0, np.dot(in_line, dense_source.unit_direction), atol=1e-7
        )

    def test_coordinates_aperture(self, dense_source):
        """Verify that the actual aperture matches the specified aperture.

        For the phased array source, the aperture should match the distance between
        edges.
        """
        coords = dense_source.coordinates
        length = np.linalg.norm(coords[-1, :] - coords[0, :])
        np.testing.assert_allclose(length, dense_source.aperture)

    def test_calculate_waveform_scale(self):
        """Verify that calculate_waveform_scale returns the expected scale factor."""
        num_elements = 5
        pitch = 0.01
        element_width = 0.009
        spacing = pitch - element_width
        length = num_elements * pitch - spacing
        num_points = 1000
        dx = 0.01
        source = TestPhasedArraySource2D.create_test_source(
            num_points=num_points, num_elements=5, pitch=0.01, element_width=0.009
        )
        scale = source.calculate_waveform_scale(dx=dx)
        grid_density = 1 / dx
        point_density = num_points / length
        np.testing.assert_allclose(scale, grid_density / point_density)

    def test_distribute_points_per_line(self):
        """Verify that source points are correctly distributed across elements."""
        num_elements = 5
        num_points = 52
        expected_points_per_el = 10
        warnings.simplefilter("ignore")
        source = TestPhasedArraySource2D.create_test_source(
            num_points=num_points, num_elements=num_elements
        )
        pm = source._distribute_points_in_elements(num_elements, num_points)
        fake_points = range(num_points)
        assert all([len(fake_points[s]) == expected_points_per_el for s in pm])

    def test_distribute_points_per_line_correct_last_index(self):
        """Verify that the last index matches `num_points`."""
        num_points = 10
        num_elements = 5
        source = TestPhasedArraySource2D.create_test_source(
            num_points=num_points, num_elements=num_elements
        )
        pm = source._distribute_points_in_elements(num_elements, num_points)
        assert pm[-1].stop == source.num_points

    def test_element_positions_property(self):
        """Verify that .elements_positions has the expected shape and values"""
        source = TestPhasedArraySource2D.create_test_source(
            num_elements=3,
            num_points=12,
            position=np.array((1.0, 2.0)),
            direction=np.array((0, -1)),
            pitch=1,
            element_width=1,
        )
        assert source.element_positions.shape == (3, 2)
        np.testing.assert_allclose(
            source.element_positions, np.array(([0, 2], [1, 2], [2, 2])), atol=1e-12
        )

    def test_distribute_points_per_line_even_distribution(self):
        """Verify that source points are correctly distributed across elements."""
        num_points = 50
        num_elements = 5
        expected_points_per_el = 10
        source = TestPhasedArraySource2D.create_test_source(
            num_points=num_points, num_elements=num_elements
        )
        pm = source._distribute_points_in_elements(num_elements, num_points)
        dif = [(s.stop - s.start) == expected_points_per_el for s in pm]
        assert all(dif)

    @pytest.mark.parametrize(
        "tilt_angle, expected_delay",
        [(30.0, 1.0), (-30.0, -1.0), (0.0, 0.0), (180.0, 0.0)],
    )
    def test_txdelay(self, tilt_angle, expected_delay):
        """Verify that delays are defined correctly."""
        delay = PhasedArraySource2D.txdelay(tilt_angle=tilt_angle, pitch=5.0, speed=2.5)
        np.testing.assert_allclose(delay, expected_delay, atol=1e-10)

    def test_calculate_focus_tilt_elements_delays_no_tilt(self):
        """Verify that delays are defined correctly only focusing."""

        source = TestPhasedArraySource2D.create_test_source(
            num_elements=3,
            num_points=3,
            pitch=1,
            element_width=1,
            focal_length=1,
            tilt_angle=0,
        )

        delays = source._calculate_focus_tilt_element_delays(speed=np.sqrt(2) - 1)
        np.testing.assert_allclose(delays, (-1, 0, -1), atol=1e-4)

    def test_calculate_focus_tilt_point_source_delays_tilt_and_focus(self):
        """Verify that delays are defined correctly only focusing."""

        source = TestPhasedArraySource2D.create_test_source(
            num_elements=3,
            num_points=3,
            pitch=1,
            element_width=1,
            focal_length=np.sqrt(2),
            tilt_angle=45,
        )

        delays = source._calculate_focus_tilt_element_delays(speed=1)
        np.testing.assert_allclose(
            delays, (-(np.sqrt(5) - 1), -(np.sqrt(2) - 1), 0), atol=1e-4
        )

    @pytest.mark.parametrize(
        "tilt_angle, focal_length",
        [(30.0, np.inf), (-30.0, 0.05), (0.0, np.inf), (180.0, 0.02)],
    )
    def test_calculate_point_source_delays_positive_delays(
        self, tilt_angle, focal_length
    ):
        """Verify that all delays are positive."""
        source = TestPhasedArraySource2D.create_test_source(
            tilt_angle=tilt_angle, focal_length=focal_length
        )
        assert np.min(source.point_source_delays) >= 0

    def test_calculate_point_source_delays_tilt_time_order(self):
        """Verify that point delays have the correct order."""
        # positive angles, correct orientation
        source = TestPhasedArraySource2D.create_test_source(
            num_elements=5, direction=np.array([1.0, 0]), tilt_angle=30.0
        )
        assert source.point_source_delays[0] < source.point_source_delays[-1]
        # negative angles, correct orientation
        source = TestPhasedArraySource2D.create_test_source(
            num_elements=5, direction=np.array([1.0, 0]), tilt_angle=-30.0
        )
        assert source.point_source_delays[0] > source.point_source_delays[-1]
        # bottom source, shouldn't change behavior.
        source = TestPhasedArraySource2D.create_test_source(
            num_elements=5, direction=np.array([-1.0, 0]), tilt_angle=-30.0
        )
        assert source.point_source_delays[0] > source.point_source_delays[-1]

    def test_calculate_point_source_delays_focus_non_focused(self):
        """Verify that delays are zero when the array should not focus."""
        source = TestPhasedArraySource2D.create_test_source(
            focal_length=np.inf, tilt_angle=0
        )
        assert all(np.isclose(source.point_source_delays, 0.0))

    def test_calculate_point_source_delays_focus_are_symmetric_even(self):
        """Verify that delays are symmetric when focusing without tilt."""
        source = TestPhasedArraySource2D.create_test_source(
            num_elements=10,
            focal_length=0.02,
            tilt_angle=0,
        )
        delays = source.point_source_delays
        assert all(np.isclose(delays - delays[::-1], 0))

    def test_calculate_point_source_delays_focus_are_symmetric_odd(self):
        """Verify that delays are symmetric when focusing without tilt."""
        source = TestPhasedArraySource2D.create_test_source(
            num_elements=5,
            focal_length=0.02,
            tilt_angle=0,
        )
        delays = source.point_source_delays
        assert all(np.isclose(delays - delays[::-1], 0))

    def test_calculate_point_source_delays_focus_symmetrical_and_right_order(self):
        """Verify that delays from tilt and focus interact correctly.

        The test creates two sources with opposite tilt angle. The obtained delays
        should be identical, except that in reverse order.
        """
        source1 = TestPhasedArraySource2D.create_test_source(
            num_elements=5,
            focal_length=0.02,
            tilt_angle=45,
        )
        source2 = TestPhasedArraySource2D.create_test_source(
            num_elements=5,
            focal_length=0.02,
            tilt_angle=-45,
        )
        # same source point distribution for both sources
        point_mapping = source1.point_mapping
        np.testing.assert_allclose(
            source2.point_source_delays[::-1], source1.point_source_delays
        )

        first_elem_delays_source_1 = source1.point_source_delays[point_mapping[0]]
        first_elem_delays_source_2 = source2.point_source_delays[point_mapping[0]]
        assert all(first_elem_delays_source_1 < first_elem_delays_source_2)

        last_elem_delays_source_1 = source1.point_source_delays[point_mapping[-1]]
        last_elem_delays_source_2 = source2.point_source_delays[point_mapping[-1]]
        assert all(last_elem_delays_source_1 > last_elem_delays_source_2)

    def test_calculate_point_source_delays_focus_distance_consistency(self):
        """Verify that delays have the expected order."""
        source = TestPhasedArraySource2D.create_test_source(
            num_elements=10,
            num_points=10,
            focal_length=1.0,
            direction=np.array([0, 1]),
            tilt_angle=-60,
            position=np.array([2, 3]),
        )
        assert source.point_source_delays.min() == 0
        delays_diff = np.diff(source.point_source_delays)
        # it should be monotonically increasing
        assert np.all(delays_diff < 0)

    @pytest.mark.parametrize(
        "unit_direction, expected",
        [
            (np.array([0, 1]), np.array([-2, 0])),
            (np.array([1, 0]), np.array([0, 2])),
            (np.array([-1, 0]), np.array([0, -2])),
            (np.array([0, -1]), np.array([2, 0])),
            (np.array([1, -1]), np.array([np.sqrt(2), np.sqrt(2)])),
            (np.array([-1, -1]), np.array([np.sqrt(2), -np.sqrt(2)])),
        ],
    )
    def test_rotate(self, unit_direction, expected):
        """Verify that the rotate function works properly for all angle spans."""
        coords = np.array([[0, 0], [2, 0]])
        rotated_coords = PhasedArraySource2D._rotate(coords, unit_direction)
        rotated_coords_dir = rotated_coords[-1, :] - rotated_coords[0, :]
        np.testing.assert_allclose(rotated_coords_dir, expected, atol=1e-10)

    def test_set_element_delays_is_none(self):
        """Verify that None `element_delays` is translated to zeros"""
        source = TestPhasedArraySource2D.create_test_source(
            num_elements=5, tilt_angle=0.0, element_delays=None
        )
        np.testing.assert_allclose(
            source.element_delays, np.zeros(shape=(5,)), atol=1e-16
        )

    def test_set_element_delays_returns_array(self):
        """Verify that `element_delays` is a 1D numpy array with shape `num_elements`"""
        source = TestPhasedArraySource2D.create_test_source(
            num_elements=10,
            tilt_angle=0,
            element_delays=list(range(0, 10)),
        )
        assert isinstance(source.element_delays, np.ndarray)
        assert source.element_delays.shape == (10,)

    def test_broadcast_delays_returns_correct_shape(self):
        """Verify that broadcast_delays returns the right shape and distribution."""
        source = TestPhasedArraySource2D.create_test_source(
            num_elements=2, num_points=10
        )
        point_source_delays = source._broadcast_delays(np.array([0.1, 0.2]))
        assert point_source_delays.shape == (10,)
        assert all(point_source_delays[0:5] == 0.1)
        assert all(point_source_delays[5:10] == 0.2)


class TestPhasedArraySource3D(Source3DTestMixin):
    @staticmethod
    def create_test_source(
        *,
        position: np.ndarray = np.array([1.0, 2.0, 3.0]),
        direction: np.ndarray = np.array([1.0, 0.0, 0.0]),
        center_line: np.ndarray = np.array([0.0, 0.0, 1.0]),
        num_points: int = 1000,
        pitch: float = 0.01,
        num_elements: int = 5,
        tilt_angle: float = 60.0,
        element_width: float = 0.009,
        height: float = 0.02,
        focal_length: float = np.inf,
        delay: float = 0,
        element_delays: np.ndarray | None = None,
    ) -> PhasedArraySource3D:
        """Creates a PhasedArraySource3D source with default parameters.

        This utility function gives default values for each parameter
        and allows us to choose to only specify one or a few parameters
        at a time.

        See `neurotechdevkit.sources.PhasedArraySource3D` for more details.

        Args:
            position: an array indicating the coordinates of the center of the source.
            direction: an array indicating the orientation of the source.
            center_line: an array indicating the direction that form the center of the
                elements.
            num_points: the number of point sources to use when simulating the source.
            pitch: the distance between the center of neighboring elements.
            num_elements: the number of elements of the phased array.
            tilt_angle: the desired tilt angle of the wavefront.
            element_width: the width of each element of the array.
            height: the height of the elements.
            focal_length: the focal length of the array.
            delay: the delay (in seconds) for the whole source.
            element_delays: the delay (in seconds) for each element of the phased array.
        Returns:
            A PhasedArraySource23 instance with the requested parameters.
        """
        warnings.simplefilter("ignore")
        return PhasedArraySource3D(
            position=position,
            direction=direction,
            center_line=center_line,
            num_points=num_points,
            pitch=pitch,
            num_elements=num_elements,
            tilt_angle=tilt_angle,
            element_width=element_width,
            height=height,
            focal_length=focal_length,
            delay=delay,
            element_delays=element_delays,
        )

    @staticmethod
    @pytest.fixture(scope="module")
    def dense_source():
        """Creates a PhasedArraySource3D with a large number of source points.

        A dense set of source points means that the points will extend very near
        to the edge of each element. This is useful for tests that verify
        dimensions of the source point line.

        This fixture has a module scope because it takes a couple of seconds to build
        the dense source point cloud.

        Returns:
            A PhasedArraySource3D instance with a dense source point cloud.
        """
        return TestPhasedArraySource3D.create_test_source(
            num_points=20_000,
        )

    def test_aperture_property(self):
        """Verify that .aperture is constructed correctly.
        Note that for PhasedArraySource3D aperture must be computed.
        """
        source = self.create_test_source(pitch=1, num_elements=10, element_width=0.9)
        # spacing = pitch - element_width
        # aperture = pitch * num_elements - spacing
        assert source.aperture == 9.9

    def test_num_points_property(self):
        """Verify that .num_points matches the value received in the constructor."""
        source = self.create_test_source(num_points=13 * 10, num_elements=13)
        assert source.num_points == 130

    def test_height_property(self):
        """Verify that .num_points matches the value received in the constructor."""
        source = self.create_test_source(height=11.0)
        assert source.height == 11.0

    def test_coordinates_has_correct_shape(self):
        """Verify that coordinates has the expected shape."""
        warnings.simplefilter("ignore")
        source = self.create_test_source(num_points=1205, num_elements=12)
        assert source.coordinates.shape == (1200, 3)

    def test_unit_center_line_property(self):
        """Verify that .unit_center_line property is set up correctly."""
        center_line = np.array([1.0, 2.0, 0])
        unit_center_line = center_line / np.linalg.norm(center_line)
        source = self.create_test_source(
            center_line=center_line, direction=np.array([0, 0, 1])
        )
        assert np.isclose(np.linalg.norm(source.unit_center_line), 1)
        np.testing.assert_allclose(np.dot(unit_center_line, source.unit_center_line), 1)

    def test_unit_center_line_raises_error_if_parallel(self):
        """Verify that center lines are not parallel to direction."""
        with pytest.raises(ValueError):
            self.create_test_source(
                direction=np.array([1, 0, 0]), center_line=np.array([1, 0, 0])
            )

    def test_unit_center_line_raises_error_if_antiparallel(self):
        """Verify that center lines are not anti parallel to direction."""
        with pytest.raises(ValueError):
            self.create_test_source(
                direction=np.array([1, 0, 0]), center_line=np.array([-1, 0, 0])
            )

    def test_coordinates_position(self, dense_source):
        """Verify that the coordinates agree with the specified position.

        The center should be the at mean of the points.
        """
        center_coords = np.mean(dense_source.coordinates, axis=0)
        np.testing.assert_allclose(center_coords, dense_source.position, atol=1e-7)

    def test_point_source_delays_property(self):
        """Verify that .point_source_delays are set up correctly in the constructor."""
        source = self.create_test_source(
            num_points=10,
            tilt_angle=0.0,
        )
        np.testing.assert_allclose(
            source.point_source_delays, np.zeros(shape=10), atol=1e-16
        )

    def test_element_delays_property(self):
        """Verify that .element_delays property is set up correctly"""
        source = self.create_test_source(
            num_points=10, num_elements=5, tilt_angle=0, element_delays=np.arange(0, 5)
        )
        np.testing.assert_array_equal(source.element_delays, np.array([0, 1, 2, 3, 4]))

    @pytest.mark.parametrize(
        "delay, fl, tilt_angle, element_delays",
        [
            (11, 12, 13, np.array([1, 2, 3])),
            (0, 12, 13, np.array([1, 2, 3])),
            (0, np.inf, 13, np.array([1, 2, 3])),
            (1, np.inf, 0, np.array([1, 2, 3])),
        ],
    )
    def test_validate_input_configuration_raises_errors(
        self, delay, fl, tilt_angle, element_delays
    ):
        """Verify that the input configuration raises value errors appropriately."""
        with pytest.raises(ValueError):
            self.create_test_source(
                tilt_angle=tilt_angle,
                delay=delay,
                focal_length=fl,
                element_delays=element_delays,
            )

    def test_element_positions_property(self):
        """Verify that .elements_positions has the expected shape and values"""
        source = self.create_test_source(
            num_elements=2,
            num_points=10,
            position=np.array((1.0, 2.0, 3.0)),
            direction=np.array((-1.0, 0, 0)),
            pitch=1,
            element_width=1,
        )
        assert source.element_positions.shape == (2, 3)
        np.testing.assert_allclose(
            source.element_positions, np.array(([1, 2, 2.5], [1, 2, 3.5]))
        )

    @pytest.mark.parametrize(
        "element_delays", [np.array([1, 2]), np.array([-1, 2, 3]), np.array([0, 0, 0])]
    )
    def test_validate_element_delays(self, element_delays):
        """Verify that argument element_delays is validated correctly."""
        with pytest.raises(ValueError):
            self.create_test_source(
                tilt_angle=0.0,
                num_elements=3,
                delay=0.0,
                focal_length=np.inf,
                element_delays=element_delays,
            )

    @pytest.mark.parametrize(
        "direction, expected",
        [
            (
                np.array([0.0, 1, 0.0]),
                np.array([0, np.cos(np.pi / 6), np.sin(np.pi / 6)]),
            ),
            (
                np.array([-np.cos(np.pi / 6), 0, 0]),
                np.array([-np.cos(np.pi / 6), 0, np.sin(np.pi / 6)]),
            ),
            (np.array([-1, 1, 0]), np.array([-0.61237244, 0.61237244, 0.5])),
        ],
    )
    def test_focal_point_is_computed_correctly(self, direction, expected):
        """Verify that focal point rotations are computed correctly."""
        source = self.create_test_source(
            position=np.array([0, 0, 0]),
            center_line=np.array([0.0, 0.0, 1.0]),
            direction=direction,
            focal_length=1,
            tilt_angle=30,
        )
        np.testing.assert_allclose(source.focal_point, expected)

    def test_focal_point_is_translates_correctly(self):
        """Verify that focal point is translated correctly in space."""
        source = self.create_test_source(
            position=np.array([1, 0, 2]),
            center_line=np.array([0.0, 0.0, 1.0]),
            direction=np.array([0.0, 1.0, 0.0]),
            focal_length=np.sqrt(2),
            tilt_angle=-45,
        )
        np.testing.assert_allclose(source.focal_point, np.array([1, 1, 1]))

    def test_distribute_points_within_element_output_shape(self, dense_source):
        """Test that the output shape is correct."""
        n_points = 20
        points = dense_source._distribute_points_within_element(10, 20, 0.1, n_points)
        assert points.shape == (n_points, 3)

    def test_distribute_points_within_element_range(self, dense_source):
        """Test that all points generated are within the specified range."""
        x_min = 11.0
        x_max = 21.0
        n_points = 200
        points = dense_source._distribute_points_within_element(
            x_min, x_max, dense_source.height, n_points
        )

        assert np.all(points[:, 0] >= x_min)
        assert np.all(points[:, 0] <= x_max)
        assert np.all(points[:, 1] >= 0.0)
        assert np.all(points[:, 1] <= dense_source.height)

    def test_distribute_points_within_element_remaining_points(self, dense_source):
        """Test that the function distributes remaining points correctly."""
        x_min = 0.0
        x_max = 10.0
        n_points = 19
        points = dense_source._distribute_points_within_element(
            x_min, x_max, dense_source.height, n_points
        )
        assert points.shape[0] == n_points
        assert np.all(points[:, 0] >= x_min)
        assert np.all(points[:, 0] <= x_max)
        assert np.all(points[:, 1] >= 0.0)
        assert np.all(points[:, 1] <= dense_source.height)

    @pytest.mark.parametrize(
        "unit_direction, expected",
        [
            (np.array([0, 1, 0]), np.array([10, 0, 0])),
            (np.array([1, 0, 0]), np.array([0, 0, 10])),
            (np.array([-1, 0, 0]), np.array([0, 0, -10])),
            (np.array([0, -1, 0]), np.array([10, 0, 0])),
            (
                np.array([-1 / np.sqrt(2), -1 / np.sqrt(2), 0]),
                np.array([10 / 2, -10 / 2, -np.sqrt(100 / 2)]),
            ),
            (np.array([0, 0, 1]), np.array([10, 0, 0])),
        ],
    )
    def test_orient(self, unit_direction, expected):
        """Verify that the rotate function works properly for all angle spans."""
        coords = np.array([[0, 0, 0], [10, 0, 0]])
        rotated_coords = PhasedArraySource3D._orient(coords, unit_direction)
        rotated_coords_dir = rotated_coords[-1, :] - rotated_coords[0, :]
        np.testing.assert_allclose(rotated_coords_dir, expected, atol=1e-10)

    def compute_observed_centerline(self, source):
        """Auxiliary function to compute actual center line."""
        coords = source.coordinates
        dst = source.point_mapping
        first_elem_center = coords[dst[0]].mean(axis=0)
        last_elem_center = coords[dst[-1]].mean(axis=0)
        arr_centerline = last_elem_center - first_elem_center
        return arr_centerline / np.linalg.norm(arr_centerline)

    @pytest.mark.parametrize(
        "center_line, direction",
        [
            [[0, 1, 1], [-1, 0, 0]],
            [[0, 1, 1], [1, 0, 0]],
            [[0, 0, 1], [0, 1, 0]],
            [[1, 1, -3], [1, -1, 0]],
            [[1, 1, -3], [0, 1, 0]],
            [[0, 1, 0], [-0.2, -0.2, -1.0]],
            [[0, -1, 0], [0.5, 0.1, 0.8]],
            [[0, -1, 0], [-0.2, -0.2, -1.0]],
            [[0, 1, 0], [0.5, 0.1, 0.8]],
            [[-1, 0, 0], [0.5, 0.1, 0.8]],
            [[0, 0, -1], [0.5, 0.1, 0.8]],
            [[0, 0, 1], [-0.1, 0.8, 0.6]],
            [[1, 1, -3], [-1, 0, -1]],
            [[1, 1, -3], [-1, -1, 0]],
            [[1, 1, 0], [-0.2, -0.2, -1.0]],
            [[1, 1, -3], [-0.1, 0.8, 0.6]],
            [[1, 1, -3], [0, 0, 1]],
            [[1, 1, -3], [-1, 0, 1]],
            [[1, 1, -3], [-0.2, -0.2, -1.0]],
            [[1, 1, -3], [0.5, 0.1, 0.8]],
        ],
    )
    def test_align_center_line_correctly_aligns(self, center_line, direction):
        """Verify that align unit center line works as expected."""
        warnings.simplefilter("ignore")
        try:
            source = self.create_test_source(
                num_elements=3,
                num_points=3,
                direction=direction,
                center_line=center_line,
            )

            cl = self.compute_observed_centerline(source)

            assert np.isclose(np.dot(source.unit_center_line, cl), 1.00)
        except ValueError:
            # ValueError is raised when direction and central line are parallel
            pass

    def test_calculate_coordinates_shape_even(self):
        """Verify that coordinates have the correct shape."""
        source = self.create_test_source(num_elements=2, num_points=10)
        assert source.coordinates.shape == (10, 3)

    def test_calculate_coordinates_shape_not_even(self):
        """Verify that coordinates have the correct shape.
        Number of points will be truncated so there is the same number of points
        per element.
        """
        warnings.simplefilter("ignore")
        source = self.create_test_source(num_elements=2, num_points=11)
        assert source.coordinates.shape == (10, 3)

    def test_calculate_coordinates_respects_height(self):
        """Verify that coordinates respect the specified height."""
        source = self.create_test_source(
            num_elements=2,
            num_points=1000,
            position=np.array([0, 0, 0]),
            direction=np.array([1, 0, 0]),
            center_line=np.array([0, 0, 1]),
        )
        assert np.all(np.abs(source.coordinates[:, 1]) <= source.height / 2)

    def test_calculate_coordinates_orientation(self):
        """Verify that coordinates show the correct orientation."""
        source = self.create_test_source(
            num_elements=2,
            num_points=10,
            position=np.array([0, 0, 0]),
            direction=np.array([0, 1, 0]),
            center_line=np.array([1, 0, 0]),
        )
        np.testing.assert_approx_equal(source.coordinates[:, 1].max(), 0)

    def test_calculate_coordinates_center_alignment(self):
        """Verify that coordinates show the correct center alignment."""
        source = self.create_test_source(
            num_elements=2,
            num_points=10_000,
            position=np.array([0, 0, 0]),
            direction=np.array([0, 0, 1]),
            center_line=np.array([0, 1, 0]),
        )
        y_width = source.coordinates[:, 1].max() - source.coordinates[:, 1].min()
        np.testing.assert_approx_equal(y_width, source.aperture)

    def test_waveform_scale_is_correct(self):
        """Verify that calculate_waveform_scale returns the expected scale factor."""
        num_elements = 5
        pitch = 0.01
        element_width = 0.009
        spacing = pitch - element_width
        length = num_elements * pitch - spacing
        num_points = 1000
        dx = 0.01
        height = 0.02
        source = self.create_test_source(
            num_points=num_points,
            num_elements=5,
            pitch=0.01,
            element_width=0.009,
            height=height,
        )
        scale = source.calculate_waveform_scale(dx=dx)
        grid_density = 1 / dx**2
        point_density = num_points / (length * height)
        np.testing.assert_allclose(scale, grid_density / point_density)

    def test_waveform_scale_is_correct_not_even_number(self):
        """Verify that calculate_waveform_scale returns the expected scale factor.

        When the number of points can not be evenly distributed in the number of
        elements, some points are discarded. The scaling needs to react appropriately.
        """
        warnings.simplefilter("ignore")
        num_elements = 10
        pitch = 0.01
        element_width = 0.009
        spacing = pitch - element_width
        length = num_elements * pitch - spacing
        num_points = 91
        quotient, _ = divmod(num_points, num_elements)
        dx = 0.01
        height = 0.02
        source = self.create_test_source(
            num_points=num_points,
            num_elements=10,
            pitch=0.01,
            element_width=0.009,
            height=height,
        )
        scale = source.calculate_waveform_scale(dx=dx)
        grid_density = 1 / dx**2
        point_density = quotient * num_elements / (length * height)
        np.testing.assert_allclose(scale, grid_density / point_density)


class TestPointSource2D:
    def test_init_properties(self):
        """Verify that .position and .delay match the value received in the
        constructor.
        """
        position = np.array([-2.0, 3.0])
        delay = 123
        source = PointSource2D(position=position, delay=delay)
        np.testing.assert_allclose(source.position, position)
        assert source.delay == delay

    def test_calculate_coordinates(self):
        """Verify that the calculated coordinates match the specified position."""
        position = np.array([-2.0, 3.0])
        source = PointSource2D(position=position)
        np.testing.assert_allclose(
            source._calculate_coordinates(), np.array([position])
        )

    def test_calculate_waveform_scale(self):
        """Verify that calculate_waveform_scale returns 1."""
        dx = 0.01
        source = PointSource2D(position=np.array([-2.0, 3.0]))
        assert source.calculate_waveform_scale(dx=dx) == 1


class TestPointSource3D:
    def test_init_properties(self):
        """Verify that .position and .delay match the value received in the
        constructor.
        """
        position = np.array([1.0, -2.0, 3.0])
        delay = 123
        source = PointSource3D(position=position, delay=delay)
        np.testing.assert_allclose(source.position, position)
        assert source.delay == delay

    def test_calculate_coordinates(self):
        """Verify that the calculated coordinates match the specified position."""
        position = np.array([1.0, -2.0, 3.0])
        source = PointSource3D(position=position)
        np.testing.assert_allclose(
            source._calculate_coordinates(), np.array([position])
        )

    def test_calculate_waveform_scale(self):
        """Verify that calculate_waveform_scale returns 1."""
        dx = 0.01
        source = PointSource3D(position=np.array([1.0, -2.0, 3.0]))
        assert source.calculate_waveform_scale(dx=dx) == 1
