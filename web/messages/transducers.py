"""Pydantic models for the transducers and their settings."""
from enum import Enum
from typing import Dict, List, Literal, Optional, Type, Union

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


class _BaseSourceSettings(BaseModel):
    @classmethod
    def from_source(cls, source) -> BaseModel:
        """Instantiate the source settings from a source."""
        raise NotImplementedError


class PointSourceSettings(_BaseSourceSettings):
    """Settings for a point source transducer."""

    transducerType: Literal["pointSource"]
    delay: float
    position: List[float]

    @classmethod
    def from_source(cls, source: PointSource2D) -> "PointSourceSettings":
        """Instantiate the source settings from a source."""
        return cls(
            transducerType="pointSource", position=source._position, delay=source._delay
        )

    def to_ndk_source(self) -> PointSource2D:
        """Instantiate the NDK source from transducer settings."""
        return PointSource2D(
            position=self.position,
            delay=self.delay,
        )


class PhasedArraySettings(_BaseSourceSettings):
    """Settings for a phased array transducer."""

    transducerType: Literal["phasedArraySource"]
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
            transducerType="phasedArraySource",
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

    transducerType: Literal["focusedSource"]
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
            transducerType="focusedSource",
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

    transducerType: Literal["planarSource"]
    aperture: float
    delay: float
    direction: List[float]
    numPoints: int
    position: List[float]

    @classmethod
    def from_source(cls, source: PlanarSource2D) -> "PlanarSourceSettings":
        """Instantiate the source settings from a source."""
        return cls(
            transducerType="planarSource",
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
