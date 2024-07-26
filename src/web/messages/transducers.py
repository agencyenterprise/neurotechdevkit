"""Pydantic models for the transducers and their settings."""
import re
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import Annotated

from neurotechdevkit.sources import (
    FocusedSource2D,
    FocusedSource3D,
    PhasedArraySource2D,
    PhasedArraySource3D,
    PlanarSource2D,
    PlanarSource3D,
    PointSource2D,
    PointSource3D,
    Source,
)


def title_case(string: str) -> str:
    """Convert a string to title case.

    Example:
        >>> title_case('helloWorld')
        'Hello World'

    Args:
        string: The string to convert to title case.

    Returns:
        The string in title case.
    """
    if string != "":
        result = re.sub("([A-Z])", r" \1", string)
        return result[:1].upper() + result[1:]
    return ""


class TransducerType(str, Enum):
    """Transducer type enum for the type of transducer."""

    pointSource = "pointSource"
    phasedArraySource = "phasedArraySource"
    focusedSource = "focusedSource"
    planarSource = "planarSource"

    @classmethod
    def from_source(
        cls,
        source: Union[
            PointSource2D,
            PointSource3D,
            PhasedArraySource2D,
            PhasedArraySource3D,
            FocusedSource2D,
            FocusedSource3D,
            PlanarSource2D,
            PlanarSource3D,
        ],
    ):
        """Instantiate the transducer type from a source."""
        if isinstance(source, (PointSource2D, PointSource3D)):
            return cls.pointSource
        elif isinstance(source, (PhasedArraySource2D, PhasedArraySource3D)):
            return cls.phasedArraySource
        elif isinstance(source, (FocusedSource2D, FocusedSource3D)):
            return cls.focusedSource
        elif isinstance(source, (PlanarSource2D, PlanarSource3D)):
            return cls.planarSource

    @classmethod
    def get_transducer_titles(cls) -> List[Tuple[str, str]]:
        """Get the transducer types and titles.

        Returns:
            The list of transducer types and titles.
        """
        titles = []
        for transducer_type in cls:
            title = title_case(transducer_type.value)
            titles.append((transducer_type.name, title))
        return titles


class _BaseSourceSettings(BaseModel):
    @classmethod
    def from_source(cls, source) -> BaseModel:
        """Instantiate the source settings from a source."""
        raise NotImplementedError


class PointSourceSettings(_BaseSourceSettings):
    """Settings for a point source transducer."""

    transducerType: Literal[TransducerType.pointSource]
    delay: float
    position: List[float]

    @classmethod
    def from_source(
        cls, source: Union[PointSource2D, PointSource3D]
    ) -> "PointSourceSettings":
        """Instantiate the source settings from a source."""
        return cls(
            transducerType=TransducerType.pointSource,
            position=source._position,
            delay=source._delay,
        )

    def to_ndk_source(self) -> Union[PointSource2D, PointSource3D]:
        """Instantiate the NDK source from transducer settings."""
        if len(self.position) == 2:
            return PointSource2D(
                delay=self.delay,
                position=self.position,
            )
        else:
            return PointSource3D(
                delay=self.delay,
                position=self.position,
            )


class PhasedArraySettings(_BaseSourceSettings):
    """Settings for a phased array transducer."""

    transducerType: Literal[TransducerType.phasedArraySource]
    position: List[float]
    direction: List[float]
    numPoints: int
    numElements: int
    pitch: float
    elementWidth: float
    tiltAngle: float
    focalLength: Optional[float]
    delay: float
    elementDelays: Optional[List[float]]
    centerLine: Optional[List[float]]
    height: Optional[float]

    @classmethod
    def from_source(
        cls, source: Union[PhasedArraySource2D, PhasedArraySource3D]
    ) -> "PhasedArraySettings":
        """Instantiate the source settings from a source."""
        if source._focal_length == np.inf:
            focal_length = None
        else:
            focal_length = source._focal_length

        element_delays = None
        if isinstance(source._configured_element_delays, list):
            element_delays = source._configured_element_delays
        elif isinstance(source._configured_element_delays, np.ndarray):
            element_delays = source._configured_element_delays.tolist()

        return cls(
            transducerType=TransducerType.phasedArraySource,
            position=source._position,
            direction=source._direction,
            numPoints=source._num_points,
            numElements=source._num_elements,
            pitch=source._pitch,
            elementWidth=source._element_width,
            tiltAngle=source._tilt_angle,
            focalLength=focal_length,
            delay=source._delay,
            elementDelays=element_delays,
            centerLine=None,
            height=None,
        )

    def to_ndk_source(self) -> Union[PhasedArraySource2D, PhasedArraySource3D]:
        """Instantiate the NDK source from transducer settings."""
        if len(self.position) == 2:
            return PhasedArraySource2D(
                position=self.position,
                direction=self.direction,
                num_points=self.numPoints,
                num_elements=self.numElements,
                pitch=self.pitch,
                element_width=self.elementWidth,
                tilt_angle=self.tiltAngle,
                focal_length=self.focalLength or np.inf,
                delay=self.delay,
                element_delays=self.elementDelays,
            )
        else:
            assert self.centerLine is not None
            assert self.height is not None
            return PhasedArraySource3D(
                position=self.position,
                direction=self.direction,
                num_points=self.numPoints,
                num_elements=self.numElements,
                pitch=self.pitch,
                element_width=self.elementWidth,
                tilt_angle=self.tiltAngle,
                focal_length=self.focalLength or np.inf,
                delay=self.delay,
                element_delays=self.elementDelays,
                center_line=self.centerLine,
                height=self.height,
            )


class FocusedSourceSettings(_BaseSourceSettings):
    """Settings for a focused source transducer."""

    transducerType: Literal[TransducerType.focusedSource]
    position: List[float]
    aperture: float
    direction: List[float]
    focalLength: float
    numPoints: int
    delay: float

    @classmethod
    def from_source(
        cls, source: Union[FocusedSource2D, FocusedSource3D]
    ) -> "FocusedSourceSettings":
        """Instantiate the source settings from a source."""
        return cls(
            transducerType=TransducerType.focusedSource,
            position=source._position,
            direction=source._direction,
            aperture=source._aperture,
            focalLength=source._focal_length,
            numPoints=source._num_points,
            delay=source._delay,
        )

    def to_ndk_source(self) -> Union[FocusedSource2D, FocusedSource3D]:
        """Instantiate the NDK source from transducer settings."""
        if len(self.position) == 2:
            return FocusedSource2D(
                position=self.position,
                direction=self.direction,
                aperture=self.aperture,
                focal_length=self.focalLength,
                num_points=self.numPoints,
                delay=self.delay,
            )
        else:
            return FocusedSource3D(
                position=self.position,
                direction=self.direction,
                aperture=self.aperture,
                focal_length=self.focalLength,
                num_points=self.numPoints,
                delay=self.delay,
            )


class PlanarSourceSettings(_BaseSourceSettings):
    """Settings for a planar source transducer."""

    transducerType: Literal[TransducerType.planarSource]
    aperture: float
    delay: float
    direction: List[float]
    numPoints: int
    position: List[float]

    @classmethod
    def from_source(
        cls, source: Union[PlanarSource2D, PlanarSource3D]
    ) -> "PlanarSourceSettings":
        """Instantiate the source settings from a source."""
        return cls(
            transducerType=TransducerType.planarSource,
            aperture=source._aperture,
            delay=source._delay,
            direction=source._direction,
            numPoints=source._num_points,
            position=source._position,
        )

    def to_ndk_source(self) -> Union[PlanarSource2D, PlanarSource3D]:
        """Instantiate the NDK source from transducer settings."""
        if len(self.position) == 2:
            return PlanarSource2D(
                aperture=self.aperture,
                delay=self.delay,
                direction=self.direction,
                num_points=self.numPoints,
                position=self.position,
            )
        else:
            return PlanarSource3D(
                aperture=self.aperture,
                delay=self.delay,
                direction=self.direction,
                num_points=self.numPoints,
                position=self.position,
            )


TRANSDUCER_SETTINGS: Dict[Type[Source], Type[_BaseSourceSettings]] = {
    PointSource2D: PointSourceSettings,
    PointSource3D: PointSourceSettings,
    PhasedArraySource2D: PhasedArraySettings,
    PhasedArraySource3D: PhasedArraySettings,
    FocusedSource2D: FocusedSourceSettings,
    FocusedSource3D: FocusedSourceSettings,
    PlanarSource2D: PlanarSourceSettings,
    PlanarSource3D: PlanarSourceSettings,
}


Transducer = Annotated[
    Union[
        PointSourceSettings,
        PhasedArraySettings,
        FocusedSourceSettings,
        PlanarSourceSettings,
    ],
    Field(discriminator="transducerType"),
]
