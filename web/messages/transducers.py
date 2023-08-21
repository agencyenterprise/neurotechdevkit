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
    PlanarSource2D,
    PointSource2D,
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
            PhasedArraySource2D,
            FocusedSource2D,
            PlanarSource2D,
        ],
    ):
        """Instantiate the transducer type from a source."""
        if isinstance(source, PointSource2D):
            return cls.pointSource
        elif isinstance(source, PhasedArraySource2D):
            return cls.phasedArraySource
        elif isinstance(source, (FocusedSource2D, FocusedSource3D)):
            return cls.focusedSource
        elif isinstance(source, PlanarSource2D):
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
    def from_source(cls, source: PointSource2D) -> "PointSourceSettings":
        """Instantiate the source settings from a source."""
        return cls(
            transducerType=TransducerType.pointSource,
            position=source._position,
            delay=source._delay,
        )

    def to_ndk_source(self) -> PointSource2D:
        """Instantiate the NDK source from transducer settings."""
        return PointSource2D(
            position=self.position,
            delay=self.delay,
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
    focalLength: float
    delay: float
    elementDelays: Optional[List[float]]

    @classmethod
    def from_source(cls, source: PhasedArraySource2D) -> "PhasedArraySettings":
        """Instantiate the source settings from a source."""
        return cls(
            transducerType=TransducerType.phasedArraySource,
            position=source._position,
            direction=source._direction,
            num_points=source._num_points,
            num_elements=source._num_elements,
            pitch=source._pitch,
            element_width=source._element_width,
            tilt_angle=source._tilt_angle,
            focal_length=source._focal_length,
            delay=source._delay,
            element_delays=source._element_delays,
        )

    def to_ndk_source(self) -> PhasedArraySource2D:
        """Instantiate the NDK source from transducer settings."""
        return PhasedArraySource2D(
            position=self.position,
            direction=self.direction,
            num_points=self.numPoints,
            num_elements=self.numElements,
            pitch=self.pitch,
            element_width=self.elementWidth,
            tilt_angle=self.tiltAngle,
            focal_length=self.focalLength,
            delay=self.delay,
            element_delays=np.array(self.elementDelays) if self.elementDelays else None,
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
    def from_source(cls, source: FocusedSource2D) -> "FocusedSourceSettings":
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

    def to_ndk_source(self) -> FocusedSource2D:
        """Instantiate the NDK source from transducer settings."""
        return FocusedSource2D(
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
    def from_source(cls, source: PlanarSource2D) -> "PlanarSourceSettings":
        """Instantiate the source settings from a source."""
        return cls(
            transducerType=TransducerType.planarSource,
            aperture=source._aperture,
            delay=source._delay,
            direction=source._direction,
            numPoints=source._num_points,
            position=source._position,
        )

    def to_ndk_source(self) -> PlanarSource2D:
        """Instantiate the NDK source from transducer settings."""
        return PlanarSource2D(
            aperture=self.aperture,
            delay=self.delay,
            direction=self.direction,
            num_points=self.numPoints,
            position=self.position,
        )


TRANSDUCER_SETTINGS: Dict[Type[Source], Type[_BaseSourceSettings]] = {
    PointSource2D: PointSourceSettings,
    PhasedArraySource2D: PhasedArraySettings,
    FocusedSource2D: FocusedSourceSettings,
    FocusedSource3D: FocusedSourceSettings,
    PlanarSource2D: PlanarSourceSettings,
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
