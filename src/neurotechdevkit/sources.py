from __future__ import annotations

import abc
import math
import warnings
from typing import Protocol

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm
from stride.utils import geometries


class Source(abc.ABC):
    """An abstract class that represents a generic Source object.

    Sources can be 2D or 3D, which affects the shape of arrays representing coordinates
    or vectors. Sources are composed of point sources evenly distributed over the
    appropriate source geometry.

    Args:
        position (npt.NDArray[np.float_]): a numpy float array indicating the
            coordinates (in meters) of the point at the center of the source.
        direction (npt.NDArray[np.float_]): a numpy float array representing a vector
            located at position and pointing towards the focal point. Only the
            orientation of `direction` affects the source, the length of the vector has
            no affect. See the `unit_direction` property.
        aperture (float): the width (in meters) of the source.
        focal_length (float): the distance (in meters) from `position` to the focal
            point.
        num_points (int): the number of point sources to use when simulating the source.
        delay (float, optional): the delay (in seconds) that the source will wait before
            emitting. Defaults to 0.0.
    """

    def __init__(
        self,
        *,
        position: npt.NDArray[np.float_],
        direction: npt.NDArray[np.float_],
        aperture: float,
        focal_length: float,
        num_points: int,
        delay: float = 0.0,
    ) -> None:
        self._validate_delay(delay)

        self._position = position
        self._unit_direction = direction / np.linalg.norm(direction)
        self._aperture = aperture
        self._focal_length = focal_length
        self._num_points = num_points
        self._delay = delay
        self._coordinates = self._calculate_coordinates()

    @property
    def coordinates(self) -> npt.NDArray[np.float_]:
        """A 2D array containing the coordinates (in meters) of the source points.

        The length of this array along the first dimension is equal to `num_points`.
        """
        return self._coordinates

    @property
    def position(self) -> npt.NDArray[np.float_]:
        """A numpy float array indicating the position (in meters) of the source.

        The position of the source is defined as the coordinates of the point at the
        center of symmetry of the source.
        """
        return self._position

    @property
    def unit_direction(self) -> npt.NDArray[np.float_]:
        """A normalized vector indicating the orientation of the source.

        The vector is located at `position`, points towards the focal point, and has
        unit length. It points in the same direction as the `direction` parameter in
        `__init__`, except it is normalized.
        """
        return self._unit_direction

    @property
    def aperture(self) -> float:
        """The width (in meters) of the source."""
        return self._aperture

    @property
    def focal_length(self) -> float:
        """The distance (in meters) from `position` to the focal point."""
        return self._focal_length

    @property
    def num_points(self) -> int:
        """The number of point sources used to simulate the source."""
        return self._num_points

    @property
    def delay(self) -> float:
        """The delay (in seconds) for the source as a whole. `delay` should be
        non-negative.
        """
        return self._delay

    @property
    def point_source_delays(self) -> npt.NDArray[np.float_]:
        """The delay before emitting (in seconds) for each point source."""
        return np.full(shape=(self.num_points,), fill_value=self.delay)

    def _validate_delay(self, delay: float) -> None:
        """Validates that `delay` is non-negative`

        Args:
            delay (float, optional): the delay (in seconds) that the source needs
                to wait before emitting. User input parameter must be non-negative.

        Raises:
            ValueError if the source delay is negatives.
        """
        if delay < 0.0:
            raise ValueError("Invalid value for `delay`. `delay` must be non-negative")

    @abc.abstractmethod
    def _calculate_coordinates(self) -> npt.NDArray[np.float_]:
        """Calculates the coordinates of the point source cloud for the source.

        This method must be implemented by all concrete Source classes.

        Returns:
            An array containing the coordinates (in meters) of the point sources that
                make up the source.
        """
        pass

    @abc.abstractmethod
    def calculate_waveform_scale(self, dx: float) -> float:
        """Calculates the scale factor to apply to waveforms from this source.

        The scale depends on the relative density of source points vs grid points.

        This method must be implemented by all concrete Source classes.

        Args:
            dx: the separation (in meters) between gridpoints. Assumed to be the same in
                all directions.

        Returns:
            The scale factor to apply to the waveform.
        """
        pass

    def waveform(self, foo, bar, baz):
        # TODO: make it the responsibility of the source
        # to calculate its own waveform
        pass


class FocusedSource2D(Source):
    """A focused source in 2D.

    This source is shaped like an arc and has a circular focus. It is created by
    taking an arc of a circle and distributing point sources evenly along that arc.

    See https://en.wikipedia.org/wiki/Circular_arc for relevant geometrical
    calculations.
    """

    @property
    def _center_angle(self) -> float:
        """The angle (in radians) which points to the center of the arc source.

        An angle of 0 points in the +x direction and positive angles go
        counter-clockwise.

        Note that `unit_direction` points in the direction of the focus while
        `_center_angle` points from the focus to the transducer, so they are
        opposite.
        """
        return np.arctan2(-self.unit_direction[1], -self.unit_direction[0])

    @property
    def _angle_range(self) -> float:
        """The total angle (in radians) that the source extends.

        The that the arc source subtends viewed from the focal point.
        """
        return 2 * np.arcsin(self.aperture / (2 * self.focal_length))

    def _calculate_coordinates(self) -> npt.NDArray[np.float_]:
        """Calculates the coordinates of the point source cloud for the source.

        This method works by calculating points uniformly spread along an arc of a
        circle.

        Returns:
            An array containing the coordinates (in meters) of the point sources that
                make up the source.

        Raises:
            ValueError: If the aperture is larger than twice the focal length. This
                situation is not possible for circular symmetry.
        """
        if self.aperture > 2 * self.focal_length:
            raise ValueError("aperture cannot be larger than twice the focal length")

        circle_center = self.position + self.unit_direction * self.focal_length
        radius = self.focal_length

        center_angle = self._center_angle
        angle_range = self._angle_range
        angles = (
            np.linspace(-angle_range / 2, angle_range / 2, self.num_points)
            + center_angle
        )

        coords = np.zeros((self.num_points, 2))
        coords[:, 0] = radius * np.cos(angles) + circle_center[0]
        coords[:, 1] = radius * np.sin(angles) + circle_center[1]

        return coords

    def calculate_waveform_scale(self, dx: float) -> float:
        """Calculates the scale factor to apply to waveforms from this source.

        The scale is equal to the ratio between the density of grid points along a line
        and the density of source points along the arc.

        Args:
            dx: the separation between gridpoints (in meters). Assumed to be the same in
                both directions.

        Returns:
            The scale factor to apply to the waveform.
        """
        grid_point_density = 1 / dx
        source_density = self.num_points / (self._angle_range * self.focal_length)
        return grid_point_density / source_density


class FocusedSource3D(Source):
    """A focused source in 3D.

    This source is shaped like a bowl and has a spherical focus. It is created by
    taking a section of a spherical shell and distributing source points over the
    surface. Points are distributed according to Fibonacci spirals.

    See https://en.wikipedia.org/wiki/Spherical_cap for relevant geometrical
    calculations.
    """

    def _calculate_coordinates(self) -> npt.NDArray[np.float_]:
        """Calculates the coordinates of the point source cloud for the source.

        This method works by calculating points along a section of a spherical shell.
        It is built on top of the stride function `geometries.ellipsoidal` but uses
        only a single radius which turns the geometry into a spherical section.
        `geometries.ellipsoidal` returns points distributed along a Fibonacci spirals,
        which produce an even density across the source surface.

        Returns:
            An array containing the coordinates (in meters) of the point sources that
                make up the source.

        Raises:
            ValueError: If the aperture is larger than twice the focal length. This
                situation is not possible for spherical symmetry.
        """
        if self.aperture > 2 * self.focal_length:
            raise ValueError("aperture cannot be larger than twice the focal length")

        sphere_center = self.position + self.unit_direction * self.focal_length
        radius = self.focal_length

        threshold = self._calculate_threshold(self.aperture, radius)
        axis, theta = self._calculate_rotation_parameters(self.unit_direction)

        return geometries.ellipsoidal(
            num=self.num_points,
            radius=(radius, radius, radius),
            centre=sphere_center,
            threshold=threshold,
            axis=tuple(axis),  # stride requires this to not be an array
            theta=theta,
        )

    def calculate_waveform_scale(self, dx: float) -> float:
        """Calculates the scale factor to apply to waveforms from this source.

        The scale is equal to the ratio between the density of grid points in a plane
        and the density of source points along the bowl surface.

        Args:
            dx: the separation between gridpoints (in meters). Assumed to be the same in
                all 3 directions.

        Returns:
            The scale factor to apply to the waveform.
        """
        grid_point_density = 1 / dx**2
        source_density = self._calculate_source_density(
            self.aperture, self.focal_length, self.num_points
        )
        return grid_point_density / source_density

    @staticmethod
    def _calculate_source_density(
        aperture: float, radius: float, num_points: int
    ) -> float:
        """Calculates the source point density (in points / meter^2).

        The density is considered based on the number of source points divided by the
        surface area of the bowl.

        Args:
            aperture: the width of the source (in meters).
            radius: the radius of curvature of the source (in meters).
            num_points (int): the number of source points in the cloud.

        Returns:
            The source point density (in points / meter^2).
        """
        aperture_ratio = aperture / (2 * radius)
        bowl_height = radius * (1 - np.sqrt(1 - aperture_ratio**2))
        bowl_surface_area = 2 * np.pi * radius * bowl_height
        return num_points / bowl_surface_area

    @staticmethod
    def _calculate_threshold(aperture: float, radius: float) -> float:
        """Calculates the threshold value to pass to Stride's
        `geometries.ellipsoidal` utility function.

        The `threshold` is used by Stride function `geometries.ellipsoidal` and is a
        number in the range [0.0, 1.0] (inclusive) which corresponds to the percent of
        the distance along the central axis of the sphere that is excluded from the
        spherical shell. A value of 0. means points are spread over the full sphere,
        while a value of 1.0 means that no points are included on the sphere.

        As this source represents a bowl, the threshold returned by this function will
        always fall in the range [0.5, 1.0] (inclusive).

        Args:
            aperture: the width of the source (in meters).
            radius: the radius of curvature of the source (in meters).

        Returns:
            The value of the threshold which produces the source bowl with the given
                `aperture` and `radius`.
        """
        ratio_to_diameter = aperture / (2 * radius)
        return 0.5 * (1 + np.sqrt(1 - ratio_to_diameter**2))

    @staticmethod
    def _calculate_rotation_parameters(
        unit_direction: npt.NDArray[np.float_],
    ) -> tuple[npt.NDArray[np.float_], float]:
        """Calculates the rotational parameters to pass to Stride's
        `geometries.ellipsoidal` utility function.

        The bowl is originally created with the axis of symmetry along the z-axis, and
        then it is rotated by `theta` radians around `axis` to align the source along
        the desired direction.

        Args:
            unit_direction: the numpy array of shape (3,) which indicates the
                orientation of the source.

        Returns:
            The rotational parameters which produce the source bowl with the given
            `unit_direction`.

            axis: the axis of rotation
            theta: the angle of rotation (in radians) around `axis`.

        Raises:
            ValueError: if `unit_direction` is not normalized.
        """
        if not math.isclose(np.linalg.norm(unit_direction), 1.0):
            raise ValueError("direction must be a normalized vector")

        v1 = np.array([0.0, 0.0, 1.0])
        # ellipsoidal always returns a sphere section oriented along the z-axis
        v2 = -unit_direction
        v3 = np.cross(v1, v2)
        mag = np.linalg.norm(v3)

        if mag == 0.0:
            # direction is along z, so we can rotate around either x or y
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = v3 / mag

        if np.dot(v1, v2) > 0:
            theta = np.arcsin(mag)
        else:
            theta = np.pi - np.arcsin(mag)

        return axis, theta


class UnfocusedMixin:
    """A mixin class for unfocused sources.

    Automatically sets `focal_length` to `np.inf`
    """

    def __init__(
        self,
        *,
        position: npt.NDArray[np.float_],
        direction: npt.NDArray[np.float_],
        aperture: float,
        num_points: int,
        delay: float = 0.0,
    ) -> None:
        super().__init__(
            position=position,
            direction=direction,
            aperture=aperture,
            focal_length=np.inf,
            num_points=num_points,
            delay=delay,
        )  # type: ignore[call-arg]


class PlanarSource2D(UnfocusedMixin, Source):
    """A planar source in 2D.

    This source is shaped like a line segment and has no focus. The source is composed
    of `num_points` point sources evenly distributed along the line segment.
    """

    def _calculate_coordinates(self) -> npt.NDArray[np.float_]:
        """Calculates the coordinates of the point source cloud for the source.

        This method works by calculating points uniformly spread along the line
        segment.

        Returns:
            An array containing the coordinates (in meters) of the point sources that
                make up the source.
        """
        unit_line = np.array([-self.unit_direction[1], self.unit_direction[0]])
        line_parametrization = np.linspace(
            -self.aperture / 2, self.aperture / 2, self.num_points
        )
        return self.position + unit_line * np.expand_dims(line_parametrization, 1)

    def calculate_waveform_scale(self, dx: float) -> float:
        """Calculates the scale factor to apply to waveforms from this source.

        The scale is equal to the ratio between the density of grid points along a line
        and the density of source points along the line segment source.

        Args:
            dx: the separation between gridpoints (in meters). Assumed to be the same in
                both directions.

        Returns:
            The scale factor to apply to the waveform.
        """
        grid_point_density = 1 / dx
        source_density = self.num_points / self.aperture
        return grid_point_density / source_density


class PlanarSource3D(UnfocusedMixin, Source):
    """A planar source in 3D.

    This source is shaped like a disk and has no focus. It is created by defining a
    disk and distributing `num_points` point sources according to Fibonacci spirals.
    """

    def _calculate_coordinates(self) -> npt.NDArray[np.float_]:
        """Calculates the coordinates of the point source cloud for the source.

        This method works by calculating points along a disk. It is built on top of the
        stride function `geometries.disk`, which returns points distributed along a
        Fibonacci spirals, in order to produce an even density across the source
        surface.

        Returns:
            An array containing the coordinates (in meters) of the point sources that
                make up the source.
        """
        return geometries.disk(
            num=self.num_points,
            radius=self.aperture / 2.0,
            centre=self.position,
            orientation=self.unit_direction,
        )

    def calculate_waveform_scale(self, dx: float) -> float:
        """Calculates the scale factor to apply to waveforms from this source.

        The scale is equal to the ratio between the density of grid points along a
        plane and the density of source points along the disk source.

        Args:
            dx: the separation between gridpoints (in meters). Assumed to be the same in
                both directions.

        Returns:
            The scale factor to apply to the waveform.
        """
        grid_point_density = 1 / dx**2
        source_density = self.num_points / (np.pi * (self.aperture / 2) ** 2)
        return grid_point_density / source_density


class _PhasedArrayMixinProtocol(Protocol):
    """Provide type-hinting for PhasedMixin"""

    _element_delays: npt.NDArray[np.float_]

    @property
    def num_points(self) -> int:
        ...

    @property
    def num_elements(self) -> int:
        ...

    @property
    def pitch(self) -> float:
        ...

    @property
    def tilt_angle(self) -> float:
        ...

    @property
    def position(self) -> npt.NDArray[np.float_]:
        ...

    @property
    def coordinates(self) -> npt.NDArray[np.float_]:
        ...

    @property
    def point_source_delays(self) -> npt.NDArray[np.float_]:
        ...

    @property
    def delay(self) -> float:
        ...

    @property
    def focal_length(self) -> float:
        ...

    @property
    def focal_point(self) -> npt.NDArray[np.float_]:
        ...

    @property
    def point_mapping(self) -> tuple[slice, ...]:
        ...

    @property
    def element_positions(self) -> npt.NDArray[np.float_]:
        ...

    @property
    def element_delays(self) -> npt.NDArray[np.float_]:
        ...

    def _broadcast_delays(
        self, delays: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        ...

    def _set_element_delays(
        self,
        element_delays: npt.NDArray[np.float_] | None,
    ) -> npt.NDArray[np.float_]:
        ...

    @staticmethod
    def txdelay(tilt_angle: float, pitch: float, speed: float = 1500.0) -> float:
        ...

    def _calculate_tilt_element_delays(self) -> npt.NDArray[np.float_]:
        ...

    def _calculate_focus_tilt_element_delays(
        self, speed: float = 1500.0
    ) -> npt.NDArray[np.float_]:
        ...


class PhasedArrayMixin:
    """A mixin class for phased array sources.
    Args:
        position (npt.NDArray[np.float_]): a numpy float array indicating
            the coordinates (in meters) of the point at the center of the
            source, which is the point that bisects the line segment source.
        direction (npt.NDArray[np.float_]): a numpy float array representing
            a vector located at position that is perpendicular to the plane
            of the source. Only the orientation of `direction` affects the
            source, the length of the vector has no affect. See the
            `unit_direction` property.
        num_points (int): the number of point sources to use when simulating
            the source.
        num_elements: the number of elements of the phased array.
        pitch: the distance (in meters) between the centers of neighboring
            elements in the phased array.
        element_width: the width (in meters) of each individual element of the array.
        tilt_angle: the desired tilt angle (in degrees) of the wavefront. The angle is
            measured between the direction the wavefront travels and the normal to the
            surface of the transducer, with positive angles resulting in a
            counter-clockwise tilt away from the normal.
        focal_length (float): the distance (in meters) from `position` to the focal
            point.
        delay (float, optional): the delay (in seconds) that the source will wait
            before emitting.
        element_delays: an 1D array with the delays (in seconds) for each element of the
            phased array. Delays from `element_delays` take precedence; No other
            argument affected the delays (`tilt_angle`, `focal_length` or `delay`)
            would be considered. ValueError will be raised if provided values for either
            `tilt_angle`, `focal_length` or `delay` are non-default.
    """

    def __init__(
        self,
        *,
        position: npt.NDArray[np.float_],
        direction: npt.NDArray[np.float_],
        num_points: int,
        num_elements: int,
        pitch: float,
        element_width: float,
        tilt_angle: float = 0.0,
        focal_length: float = np.inf,
        delay: float = 0.0,
        element_delays: npt.NDArray[np.float_] | None = None,
    ) -> None:

        self._validate_input_configuration(
            tilt_angle=tilt_angle,
            focal_length=focal_length,
            delay=delay,
            element_delays=element_delays,
        )
        self._validate_num_elements(num_elements)
        self._validate_element_delays(element_delays, num_elements)

        self._num_elements = num_elements
        self._pitch = pitch
        self._tilt_angle = tilt_angle
        self._element_width = element_width
        num_points = self._validate_num_points(num_points)
        self._point_mapping = self._distribute_points_in_elements(
            num_elements, num_points
        )

        super().__init__(
            position=position,
            direction=direction,
            aperture=np.nan,
            focal_length=focal_length,
            num_points=num_points,
            delay=delay,
        )  # type: ignore[call-arg]

        self._element_delays = self._set_element_delays(element_delays)  # type: ignore

    @property
    def num_elements(self) -> int:
        """The number of elements in the source array."""
        return self._num_elements

    @property
    def pitch(self) -> float:
        """The pitch (in meters) of the source."""
        return self._pitch

    @property
    def aperture(self) -> float:
        """The width (in meters) of the source."""
        return self.num_elements * self.pitch - self.spacing

    @property
    def tilt_angle(self) -> float:
        """The angle (in degrees) that the wave front is tilted."""
        return self._tilt_angle

    @property
    def element_width(self) -> float:
        """The width (in meters) of each element of the array."""
        return self._element_width

    @property
    def spacing(self) -> float:
        """The separation (in meters) between elements of the array."""
        return self.pitch - self.element_width

    @property
    def point_mapping(self) -> tuple[slice, ...]:
        """A tuple with the slices of source point indexes comprising each element."""
        return self._point_mapping

    @property
    def element_delays(self) -> npt.NDArray[np.float_]:
        """The delay (in seconds) that each element should wait before emitting."""
        return self._element_delays

    @property
    def point_source_delays(self: _PhasedArrayMixinProtocol) -> npt.NDArray[np.float_]:
        """The delay before emitting (in seconds) for each point source.

        The delays are computed at the element level. All source points within an
        element will have the same delay.
        """
        return self._broadcast_delays(self.element_delays)

    @property
    def element_positions(self: _PhasedArrayMixinProtocol) -> npt.NDArray[np.float_]:
        """An array with the position of the  center of each element of the array"""

        positions = np.zeros(shape=(self.num_elements, len(self.position)))
        point_mapping = self.point_mapping
        coords = self.coordinates
        for i in range(self.num_elements):
            el_coords = coords[point_mapping[i]]
            el_coords_mean = el_coords.mean(axis=0)
            positions[i, :] = el_coords_mean[:]
        return positions

    @staticmethod
    def _validate_input_configuration(
        tilt_angle: float,
        focal_length: float,
        delay: float,
        element_delays: npt.NDArray[np.float_] | None,
    ) -> None:
        """Check that the input arguments for delays yield a valid configuration.

        Args:
            tilt_angle: angle (in degrees) between the vector normal to the
                source and the wavefront.
            focal_length (float): the distance (in meters) from `position` to
                the focal point.
            delay (float, optional): the delay (in seconds) that the source will
                wait before emitting.
            element_delays: an 1D array with the delays (in seconds) for each
                element of the phased array.

        Raises:
            ValueError if the combination of input arguments is an invalid one.
        """
        if element_delays is not None:
            others_are_default = (
                (tilt_angle == 0.0) and np.isinf(focal_length) and (delay == 0.0)
            )
            if not others_are_default:
                raise ValueError(
                    "If element_delays argument is not None, `tilt_angle`"
                    " `focal_length`, and `delay` must be left at their default value."
                )

    @staticmethod
    def _validate_num_elements(num_elements: int) -> None:
        """Ensures that the number of elements is positive, greater than 1.

        Currently phased arrays with one element are not supported.

        Args:
            num_elements: the number of elements in the phased array.

        Raises:
            ValueError if the number of elements is smaller than 2.
        """
        if num_elements < 2:
            raise ValueError(
                f"Expecting number of elements larger than 1. Got {num_elements}."
            )

    @staticmethod
    def _validate_element_delays(element_delays, num_elements) -> None:
        """Checks that the input value for `element_delays` meets the requirements.

        If `element_delays` is None, no check is performed
        If `element_delays` is not None, it must be a 1D array with length equal to
        `num_elements`, all delays should be positive and at least one delay greater
        than zero.

        Args:
            element_delays: the delays (in seconds) per elements passed by the user
                (or None, its default value)
            num_elements: the number of elements of the phased array.

        Raises:
            ValueError if `element_delays` is invalid.
        """

        if element_delays is None:
            return
        element_delays = np.array(element_delays)
        if (
            (element_delays.shape != (num_elements,))
            or (element_delays.min() < 0.0)
            or (element_delays.sum() == 0.0)
        ):
            raise ValueError(
                "Invalid value for `element_delays`",
                " `element_delays` must be of 1D array of length `num_elements`.",
                " All values must be non-negative. At least one value must be greater "
                " than 0.",
            )

    def _validate_num_points(self, num_points: int) -> int:
        """Ensure that `num_points` can be distributed evenly in `num_elements`.

        Args:
            num_points, the number of points that the user requested.

        Returns:
            the number of points that can distributed evenly in `num_elements`.
        """
        quotient, remainder = divmod(num_points, self.num_elements)
        if remainder > 0:
            warnings.warn(
                f"The number of points {num_points } has been truncated to "
                f"{self.num_elements*quotient} to be evenly distributed in "
                f"{self.num_elements} elements. {remainder} points discarded.",
                category=UserWarning,
            )
        return int(quotient * self.num_elements)

    def _calculate_line_boundaries(
        self,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
        """
        Calculates the start and end of each line element in 1D.

        Returns:
            A tuple with two arrays for the minimum and maximum position.
        """
        center_coords = (
            np.arange(self.num_elements) * self.pitch + self.element_width / 2
        )
        x_min = center_coords - self.element_width / 2
        x_max = center_coords + self.element_width / 2

        return x_min, x_max

    @staticmethod
    def _distribute_points_in_elements(
        num_elements: int, num_points: int
    ) -> tuple[slice, ...]:
        """
        Distributes `num_points` evenly among a specified number of elements.

        Computes an array indicating the start and end indices of points assigned to
        each element.

        Args:
            num_elements: the number of elements forming the array.
            num_points (int): the number of source points to distribute.

        Returns:
            A 2D numpy array of shape (num_elements, 2) indicating the start and end
            indices of the points assigned to each element.
        """
        quotient, _ = divmod(num_points, num_elements)
        points_per_element = np.full(num_elements, quotient)

        # start and end index for each element
        point_assignment = np.zeros(shape=(num_elements, 2), dtype=np.int_)
        start_index = 0
        for i, n_points in enumerate(points_per_element):
            point_assignment[i, :] = start_index, start_index + n_points
            start_index += n_points

        return tuple([slice(x[0], x[1]) for x in point_assignment])

    @staticmethod
    def _translate(
        coords: npt.NDArray[np.float_], position: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """
        Translates the source to place the center at `position`.

        Args:
            coords: array with the coordinates to translate.
            position: the center of the array.

        Returns:
            An array containing the coordinates (in meters) of the point sources that
            make up the source.
        """
        coords = coords - coords.mean(axis=0)
        coords = coords + position

        return coords

    def _broadcast_delays(
        self: _PhasedArrayMixinProtocol, delays: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """Translates the delays per element into delays per source point.

        All source points within one element have the same delay.

        Args:
            delays: the delays (in seconds) for each element of the array.

        Returns:
            the delays (in seconds) per each source point of the array.
        """
        point_source_delays = np.zeros(self.num_points)
        for i, slc in enumerate(self.point_mapping):
            point_source_delays[slc] = delays[i]
        return point_source_delays

    @staticmethod
    def txdelay(
        tilt_angle: float,
        pitch: float,
        speed: float = 1500,  # m/s speed of sound in water
    ) -> float:
        """
        Computes the delay (in seconds) required to tilt the wavefront.

        The delays from element n to element n+1 to achieve a wavefront with
        `tilt_angle` respect to the normal. Positive angles lead to counter-clockwise
        rotations.

        Args:
            tilt_angle: angle (in degrees) between the vector normal to the source
                        and the wavefront.
            pitch: the pitch (in meters) of the source.
            speed: the speed of sound (in meters/second) of the material where
                   the source is placed.

        Returns:
            the delay (in seconds) between two consecutive elements.
        """
        tilt_radians = np.radians(tilt_angle)
        phase_time = pitch * np.cos(np.pi / 2 - tilt_radians)
        phase_time = phase_time / speed

        return phase_time

    def _calculate_tilt_element_delays(
        self: _PhasedArrayMixinProtocol,
    ) -> npt.NDArray[np.float_]:
        """
        Calculate delays (in seconds) per array element to produce a given `tilt_angle`.

        Returns:
            An array with the delay (in seconds) per array element.
        """
        elementwise_delay = self.txdelay(self.tilt_angle, self.pitch)
        delays = np.arange(self.num_elements) * elementwise_delay
        return delays

    def _calculate_focus_tilt_element_delays(
        self: _PhasedArrayMixinProtocol, speed=1500.0  # m/s speed of sound in water
    ) -> npt.NDArray[np.float_]:
        """
        Calculate delays (in seconds) per array element to focus the source.

        Args:
            speed: the speed of sound (in m/s) of the material where
                the source is placed.

        Returns:
            An array with the negative delay (in seconds) per array element.
        """

        delays = np.zeros(shape=(self.num_elements,))
        distances = np.array(
            [np.linalg.norm(ec - self.focal_point) for ec in self.element_positions],
            dtype=np.float_,
        )
        extra_distance = distances - distances.min()
        delays = -extra_distance / speed

        return delays

    def _set_element_delays(
        self: _PhasedArrayMixinProtocol,
        element_delays: npt.NDArray[np.float_] | None,
    ) -> npt.NDArray[np.float_]:
        """
        Calculate delays (in seconds) per element of the phased array.

        If the user specified the `element_delays` it will take precedence and no other
        delay will be computed.

        If `element_delays` is None, it computes the delays to achieve a `tilt_angle`
        focused at `focal_length`. It also adds the global `delay` for the whole source.

        Delays are non negative.

        Args:
            element_delays: the delays (in seconds) for each element of the array or
                None.

        Returns:
            An array with the delay (in seconds) per element.

        """
        if element_delays is not None:
            element_delays = np.array(element_delays)
            return element_delays

        if np.isfinite(self.focal_length):
            delays = self._calculate_focus_tilt_element_delays()
        else:
            delays = self._calculate_tilt_element_delays()

        # make all delays non-negative
        delays += np.abs(delays.min()) + self.delay

        return delays


class PhasedArraySource2D(PhasedArrayMixin, Source):
    """A phased array source in 2D.

    This source is shaped like a multiple segments in a line. Each segment can emit
    waves independently. It has no focus currently. A focused implementation will be
    supported in the future. This source is composed of `num_points` point sources.
    Distributed evenly in `num_elements`.

    If the number of points can not be evenly distributed in the number of elements, the
    remainder number of points from the even division will be discarded.

    See https://en.wikipedia.org/wiki/Phased_array_ultrasonics for detailed explanation.
    """

    @property
    def focal_point(self) -> npt.NDArray[np.float_]:
        """The coordinates (in meters) of the point where the array focuses.

        If the array is unfocused it will return the focal point (inf, inf).
        """
        if np.isinf(self.focal_length):
            return np.array([np.inf, np.inf])

        focal_vector = self.focal_length * self.unit_direction
        angle = np.deg2rad(self.tilt_angle)
        focal_point = self.position + _rotate_2d(focal_vector, angle)
        return focal_point

    @staticmethod
    def _rotate(
        coords: npt.NDArray[np.float_], unit_direction: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """
        Rotates the source to align its normal direction to `unit_direction`.

        Args:
            coords: array with coordinates distributed along the x axis with dimension.
            unit_direction: direction that normal of the source points to
        Returns:
            an array with the rotated coordinates (in meters) of the point sources.
        """
        # compute normal to the line direction
        # line direction is by construction along the x axis
        dx, dy = 1, 0

        # vectors to align
        u = np.array([dy, -dx])
        v = unit_direction

        # rotation parameters
        angle = np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0])
        r_coords = _rotate_2d(coords, angle)

        return r_coords

    def _calculate_coordinates(self) -> npt.NDArray[np.float_]:
        """Calculates the coordinates of the point source cloud for the source.

        This method works by calculating points uniformly spread along the line
        of each segment. When `num_points` can not be evenly distributed in
        `num_elements` the reminder of points are assigned randomly to elements.

        Returns:
            An array containing the coordinates (in meters) of the point sources that
            make up the source.
        """
        x_min, x_max = self._calculate_line_boundaries()
        coords = np.zeros(shape=(self.num_points, 2))
        point_mapping = self.point_mapping

        for i in range(self.num_elements):
            n = len(coords[point_mapping[i]])
            x_coords = np.linspace(x_min[i], x_max[i], num=n)
            coords[point_mapping[i], 0] = x_coords

        # rotation and translation to match user inputs
        coords = self._rotate(coords, self.unit_direction)
        coords = self._translate(coords, self.position)

        return coords

    def calculate_waveform_scale(self, dx: float) -> float:
        """Calculates the scale factor to apply to waveforms from this source.

        The scale is equal to the ratio between the density of grid points along a line
        and the density of source points along the line segment source.

        Args:
            dx: the separation between gridpoints (in meters). Assumed to be the same in
                both directions.

        Returns:
            The scale factor to apply to the waveform.
        """
        grid_point_density = 1 / dx
        source_density = self.num_points / self.aperture

        return grid_point_density / source_density


class PhasedArraySource3D(PhasedArrayMixin, Source):
    """A linear phased array source in 3D.

    This source is shaped like a multiple rectangular segments in a line. Each segment
    can emit waves independently. It has no focus currently. A focused implementation
    will be supported in the future. This source is composed of `num_points` point
    sources distributed evenly in `num_elements`.

    If the number of points can not be evenly distributed in the number of elements, the
    remainder number of points from the even division will be discarded.

    See https://en.wikipedia.org/wiki/Phased_array_ultrasonics for detailed explanation.

    Args:
        position (npt.NDArray[np.float_]): a numpy float array in 3D indicating the
            coordinates (in meters) of the point at the center of the source, which is
            the point that bisects both the height and the aperture of the source.
        direction (npt.NDArray[np.float_]): a numpy float array in 3D representing a
            vector located at position that is perpendicular to the plane of the source.
            Only the orientation of `direction` affects the source, the length of
            the vector has no affect. See the `unit_direction` property.
        center_line: A 3D vector which is parallel to the line through the centers of
            the elements in the linear array. This vector must be perpendicular to
            `direction`. If the vector is not perpendicular, only the perpendicular
            component will be considered. Only the orientation affects the source, the
            length of the vector has no effect. See `unit_center_line` property.
        num_points (int): the number of point sources to use when simulating the source.
            If the number of points is not divisible evenly by the number of elements,
            the number of points would be truncated to a multiple of the maximum even
            divisor.
        num_elements: the number of elements of the phased array.
        pitch: the distance (in meters) between the centers of neighboring elements in
            the phased array.
        height: the height (in meters) of the elements of the array. `height` is
            measured along the direction in the plane of the element that is
            perpendicular to `center_line`.
        element_width: the width (in meters) of each individual element of the array.
        tilt_angle: the desired tilt angle (in degrees) of the wavefront. The angle is
            measured between the direction the wavefront travels and the normal to the
            surface of the transducer, with positive angles resulting in a
            counter-clockwise tilt away from the normal.
        focal_length (float): the distance (in meters) from `position` to the focal
            point.
        delay (float, optional): the delay (in seconds) that the source will wait before
            emitting.
        element_delays: an 1D array with the delays (in seconds) for each element of the
            phased array. Delays from `element_delays` take precedence; No other
            argument affected the delays (`tilt_angle`, `focal_length` or `delay`)
            would be considered. ValueError will be raised if provided values for either
            `tilt_angle`, `focal_length` or `delay` are non-default.
    """

    def __init__(
        self,
        *,
        position: npt.NDArray[np.float_],
        direction: npt.NDArray[np.float_],
        center_line: npt.NDArray[np.float_],
        num_points: int,
        num_elements: int,
        pitch: float,
        height: float,
        element_width: float,
        tilt_angle: float = 0.0,
        focal_length: float = np.inf,
        delay: float = 0.0,
        element_delays: npt.NDArray[np.float_] | None = None,
    ) -> None:

        self._height = height
        self._unit_center_line = self._validate_center_line(center_line, direction)

        super().__init__(
            position=position,
            direction=direction,
            num_points=num_points,
            num_elements=num_elements,
            pitch=pitch,
            tilt_angle=tilt_angle,
            element_width=element_width,
            focal_length=focal_length,
            delay=delay,
            element_delays=element_delays,
        )

    @property
    def height(self) -> float:
        """The height (in meters) of the elements of the source."""
        return self._height

    @property
    def unit_center_line(self) -> npt.NDArray[np.float_]:
        """The unit direction of the line crossing the center of the array elements."""
        return self._unit_center_line

    @property
    def focal_point(self) -> npt.NDArray[np.float_]:
        """The coordinates (in meters) of the point where the array focuses.

        If the array is unfocused it will return the focal point (inf, inf, inf).
        """
        if np.isinf(self.focal_length):
            return np.array([np.inf, np.inf, np.inf])

        focal_vector = self.focal_length * self.unit_direction
        axis = np.cross(self.unit_direction, self.unit_center_line)
        rotated_fp = _rotate_3d(
            coords=focal_vector, axis=axis, theta=np.deg2rad(self.tilt_angle)
        )
        focal_point = self.position + rotated_fp
        return focal_point

    @staticmethod
    def _validate_center_line(
        center_line: npt.NDArray[np.float_], direction: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """Ensures that the center line input parameter is a valid one.

        Center line must not be parallel to the direction.
        Center line must be perpendicular to the direction.

        The center line only makes sense if it is orthogonal to the direction of the
        array. If the input value is not orthogonal, it will be adjusted to keep the
        orthogonal component only.

        Args:
            center_line: an 1D array with the input value for the direction the center
                of elements should have.
            direction: an 1D array with the normal direction to the phased array.

        Returns:
            a 1D array with a valid unit center line.

        Raises:
            ValueError if center_line and direction are parallel or antiparallel.

        Warns:
            When the input center line needs to be transformed to an orthogonal unit
            center line.
        """
        center_line = _unit_vector(center_line)
        direction = _unit_vector(direction)

        dot_prod = np.dot(center_line, direction)

        # Raise error if parallel
        if np.isclose(np.abs(dot_prod), 1):
            raise ValueError(
                "Center line vector can not be parallel to the direction"
                " of the array. Bad argument for `center_line`."
            )

        if dot_prod == 0:
            return center_line
        else:
            # remove orthogonal component
            parallel_to_normal_component = dot_prod * direction
            orth_center_line = center_line - parallel_to_normal_component
            warnings.warn(
                "The received value for `center_line` is not perpendicular"
                " to `direction`. Only the perpendicular component will be"
                f" used. The updated center_line value is {orth_center_line}"
            )
            return _unit_vector(orth_center_line)

    @staticmethod
    def _distribute_points_within_element(
        x_min: float, x_max: float, height: float, n_points: int
    ) -> npt.NDArray[np.float_]:
        """Distribute a given number of points in a 3D space.

        Args:
            x_min: the minimum value of the x-axis.
            x_max: the maximum value of the x-axis.
            height: the height (in meters) of the array.
            n_points: the number of points to distribute.

        Returns:
            An array of shape (`n_points`, 3) with the coordinates of the points.
            The points are placed in the XY plane. The returned Z coordinates are zero.

        This function generates a set of uniformly distributed points in a rectangular
        area defined by the x_min and x_max parameters. It first generates a grid of
        points using the number of rows and columns calculated from the square root of
        n_points. Any remaining points are distributed in an "X" pattern, with the
        center point of the pattern being the center of the rectangular area. The points
        along the diagonal and anti-diagonal axes of the rectangle are generated and
        added to the existing grid of points.
        """

        # Calculate the number of rows and columns in the grid
        n_rows = int(np.sqrt(n_points))
        n_cols = int(np.floor(n_points / n_rows))

        # Generate a set of uniformly distributed points using the grid pattern
        x_coords = np.linspace(x_min, x_max, n_cols)
        y_coords = np.linspace(0, height, n_rows)
        xx, yy = np.meshgrid(x_coords, y_coords)
        points = np.vstack((xx.ravel(), yy.ravel())).T

        # Distribute any remaining points in an "X" pattern
        n_remaining = n_points - points.shape[0]

        if n_remaining > 0:

            # First compute the center
            x_width = x_max - x_min
            centre = np.array([(x_min + x_max) / 2, height / 2])
            NON_EVEN_NUMBER = 3.14  # minimizes risk of placing two points in same spot
            x_min_r = centre[0] - x_width / NON_EVEN_NUMBER
            x_max_r = centre[0] + x_width / NON_EVEN_NUMBER
            y_min_r = centre[1] - height / NON_EVEN_NUMBER
            y_max_r = centre[1] + height / NON_EVEN_NUMBER

            # Generate a set of points along the diagonal axes of the rectangle
            diag_x = np.linspace(x_min_r, x_max_r, n_remaining // 2 + 1)
            diag_y = np.linspace(y_min_r, y_max_r, n_remaining // 2 + 1)
            diag_points = np.vstack((diag_x, diag_y)).T

            # Add the diagonal points to the existing grid of points
            points = np.vstack((points, diag_points))
            n_remaining -= diag_points.shape[0]

            # Generate a set of points along the anti-diagonal axes of the rectangle
            anti_diag_x = np.linspace(x_min_r, x_max_r, n_remaining)
            anti_diag_y = np.linspace(y_max_r, y_min_r, n_remaining)
            anti_diag_points = np.vstack((anti_diag_x, anti_diag_y)).T

            # Add the anti-diagonal points to the existing grid of points
            points = np.vstack((points, anti_diag_points))

        # adds Z coordinate
        points = np.pad(points, ((0, 0), (0, 1)), mode="constant")

        return points

    @staticmethod
    def _orient(
        coords: npt.NDArray[np.float_], unit_direction: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """
        Orient a set of 3D coordinates relative to a given unit direction vector.

        The function first calculates a normal vector to the XY plane that is
        perpendicular to the given direction vector. It then rotates the input
        coordinates around this normal vector, so that the direction vector is aligned
        with the Z axis. The resulting coordinates are then returned as the output.
        If the direction vector is already parallel to the Z axis, the function rotates
        the coordinates around either the X or Y axis.

        Args:
            coords: An array containing the 3D coordinates to be oriented.
            unit_direction: A 1D array specifying the direction vector that the normal
                of the coordinates should be oriented towards.

        Returns:
            An array of containing the oriented 3D coordinates.
        """
        # The source is created in the XY plane, the normal direction is:
        v1 = np.array([0.0, 0.0, 1.0])
        v2 = -unit_direction
        v3 = np.cross(v1, v2)
        mag = np.linalg.norm(v3)

        if mag == 0.0:
            # direction is along z, so we can rotate around either x or y
            axis = np.array([1.0, 0.0, 0.0])
        else:
            axis = v3 / mag

        if np.dot(v1, v2) > 0:
            theta = np.arcsin(mag)
        else:
            theta = np.pi - np.arcsin(mag)

        oriented_coords = _rotate_3d(coords=coords, axis=axis, theta=theta)
        return oriented_coords

    def _align_to_unit_center_line(
        self,
        coords: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        """
        Aligns a the coordinates to the desired direction for the unit center line.

        The first and last element centers are calculated and the direction vector from
        the first element to the last element is computed.

        The current alignment is rotated to match the `self.unit_center_line` alignment.

        Args:
            coords: Aa array containing the 3D coordinates to be aligned.

        Returns:
            An array of shape containing the aligned 3D coordinates.

        """
        v2 = self.unit_center_line
        point_mapping = self.point_mapping
        first_elem_center = coords[point_mapping[0]].mean(axis=0)
        last_elem_center = coords[point_mapping[-1]].mean(axis=0)
        arr_centerline = last_elem_center - first_elem_center

        v1 = _unit_vector(arr_centerline)
        v3 = np.cross(v1, v2)
        axis = self.unit_direction
        mag = np.dot(v3, axis)

        if np.dot(v1, v2) > 0:
            theta = np.arcsin(mag)
        else:
            theta = np.pi - np.arcsin(mag)

        oriented_coords = _rotate_3d(coords=coords, axis=axis, theta=theta)

        return oriented_coords

    def _calculate_coordinates(self) -> npt.NDArray[np.float_]:
        """Calculates the coordinates of the point source cloud for the source.

        Returns:
            An array containing the coordinates (in meters) of the point sources that
            make up the source.
        """
        x_min, _ = self._calculate_line_boundaries()
        coords = np.zeros(shape=(self.num_points, 3))

        point_mapping = self.point_mapping
        N = len(coords[point_mapping[0]])

        tcrds = self._distribute_points_within_element(
            x_min=0.0, x_max=self.element_width, height=self.height, n_points=N
        )

        for i in range(self.num_elements):
            element_offset = np.array([x_min[i], 0.0, 0.0])
            coords[point_mapping[i]] = element_offset + tcrds

        # rotation and translation to match user inputs
        coords = self._orient(coords, self.unit_direction)
        coords = self._align_to_unit_center_line(coords)
        coords = self._translate(coords, self.position)
        return coords

    def calculate_waveform_scale(self, dx: float) -> float:
        """Calculates the scale factor to apply to waveforms from this source.

        The scale is equal to the ratio between the density of grid points along a
        plane and the density of source points along the planar source.

        Args:
            dx: the separation between gridpoints (in meters). Assumed to be the same in
                both directions.

        Returns:
            The scale factor to apply to the waveform.
        """
        grid_point_density = 1 / dx**2
        source_density = self.num_points / (self.aperture * self.height)
        return grid_point_density / source_density


def _rotate_2d(coords: npt.NDArray[np.float_], theta: float) -> npt.NDArray[np.float_]:
    """Rotates `coords` around the origin an angle `theta` around the origin (0, 0).
    Rotation is in 2D.

    Args:
         coords: An array with the coordinates to rotate.
         theta: the angle (in radians) specifying rotation around the origin.

    Returns:
       An array with the rotated coordinates.
    """
    coords = np.atleast_2d(coords)
    rot_mat = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    rotated = ((rot_mat @ coords.T).T).squeeze()
    return rotated


def _rotate_3d(
    coords: npt.NDArray[np.float_], axis: npt.NDArray[np.float_], theta: float
) -> npt.NDArray[np.float_]:
    """Compute the rotation matrix in 3D to rotate around `axis` an angle `theta`.

    Args:
        coords: An array with the coordinates to rotate.
        axis: A 1-D array specifying the axis, around which, the rotation would happen.
        theta: A float (in radians) specifying rotation around the axis.

    Returns:
       An array with the rotated coordinates.

    Raises:
        ValueError: If `axis` is not a 1-D array with 3 elements or if its norm is zero.
    """
    coords = np.atleast_2d(coords)
    if axis.shape != (3,):
        raise ValueError(
            f"Expected a 1-D array of length 3 for `axis`, but got {axis}."
        )
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("The norm of `axis` must be nonzero.")
    rot_mat = expm(np.cross(np.eye(3), axis / norm * theta))
    rotated = ((rot_mat @ coords.T).T).squeeze()
    return rotated


def _unit_vector(vector: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """Returns the unit vector that is parallel to the input vector.

    Args:
        vector: a 1D numpy array.

    Returns:
        A normalized array

    """
    return vector / np.linalg.norm(vector)
