from __future__ import annotations

from typing import NamedTuple, Protocol

import numpy as np
import numpy.typing as npt

# Warning: this is a circular import, so we can't access any members of
# scenario during module import. This affects type-hinting in particular,
# so forward references (in quotes) need to be used.
# This should be fixed in the future.
from .. import scenarios, sources


class _NapariViewer(Protocol):
    """A Protocol for type-hinting of napari.Viewer.

    We're using a Protocol to stub the type-hinting of napari.Viewer
    so that we don't need to import napari at the module level.
    """

    def add_image(
        self,
        data=None,
        *,
        name=None,
        rendering="mip",
        iso_threshold=0.5,
        colormap=None,
        opacity=1,
    ):
        pass

    def add_shapes(
        self,
        data=None,
        *,
        name=None,
        shape_type="rectangle",
        edge_color="#777777",
        face_color="#white",
        edge_width=1,
        opacity=0.7,
    ):
        pass

    def add_points(
        self,
        data=None,
        *,
        name=None,
        symbol="o",
        size=10,
        face_color="white",
        edge_color="dimgray",
        opacity=1,
    ):
        pass


class ViewerConfig3D(NamedTuple):
    """Configuration parameters for 3D visualization of scenarios.

    Attributes:
        init_angles: The viewing angle to set on startup.
        init_zoom: The zoom to set at startup.
        colormaps: A map from layer names to the name of the colormap which should
            be used to display the layer.
        opacities: A map from layer names to the opacity that should be used for the
            layer.
    """

    init_angles: tuple[float, float, float]
    init_zoom: float
    colormaps: dict[str, str]
    opacities: dict[str, float]


def render_layout_3d_with_napari(scenario: "scenarios.Scenario3D") -> None:
    """Render the scenario layout in 3D using napari.

    Args:
        scenario: The 3D scenario to be rendered.

    Raises:
        ImportError: If napari is not found.
    """
    _create_napari_3d(scenario=scenario, amplitudes=None)


def render_amplitudes_3d_with_napari(result: "scenarios.SteadyStateResult3D") -> None:
    """Render the scenario layout in 3D using napari.

    Args:
        scenario: The 3D scenario to be rendered.

    Raises:
        ImportError: If napari is not found.
    """
    pass
    _create_napari_3d(scenario=result.scenario, amplitudes=result.get_steady_state())


def _create_napari_3d(
    scenario: "scenarios.Scenario3D", amplitudes: npt.NDArray[np.float_] | None
) -> None:
    try:
        import napari
    except ModuleNotFoundError as e:
        raise ImportError(
            "3D rendering requires napari to be installed. Integration with napari is"
            " experimental and is not included as an explicit dependency of NDK."
            ' Please install via `pip install "napari[all]"` or follow napari\'s'
            " instructions at:"
            " https://napari.org/stable/tutorials/fundamentals/installation.html"
        ) from e

    viewer_config = scenario.viewer_config_3d
    viewer = napari.Viewer(ndisplay=3)

    add_material_layers(viewer, scenario, viewer_config)
    add_target(viewer, scenario)

    scenario._ensure_source()
    for source in scenario.sources:
        add_source(viewer, scenario, source)

    if amplitudes is not None:
        add_steady_state_amplitudes(viewer, amplitudes)

    viewer.camera.angles = viewer_config.init_angles
    viewer.camera.zoom = viewer_config.init_zoom

    print(
        "Opening the napari viewer. The window might not show up on top of your"
        " notebook; look through your open applications if it does not."
    )

    viewer.show()


def add_material_layers(
    viewer: _NapariViewer,
    scenario: "scenarios.Scenario3D",
    viewer_config: ViewerConfig3D,
) -> None:
    """Adds the individual material layers as images to a napari Viewer.

    Args:
        viewer: The napari Viewer to which the layers should be added.
        scenario: The 3D scenario which is being visualized.
        viewer_config: The configuration parameters for the 3d visualization.
    """
    colormaps = viewer_config.colormaps
    opacities = viewer_config.opacities
    layer_data = scenario.get_field_data("layer")

    for name, layer_id in scenario.layer_ids.items():
        viewer.add_image(
            name=name,
            data=layer_data == layer_id,
            rendering="iso",
            iso_threshold=0.0,
            colormap=colormaps[name],
            opacity=opacities[name],
        )


def add_steady_state_amplitudes(
    viewer: _NapariViewer, amplitudes: npt.NDArray[np.float_]
) -> None:
    """Adds the steady-state amplitudes as an image layer to a napari Viewer.

    Args:
        viewer: The napari Viewer to which the target should be added.
        amplitudes: A 3D numpy array containing the steady-state amplitudes.
    """
    viewer.add_image(
        name="steady-state amplitudes",
        data=amplitudes,
        rendering="mip",
        colormap="viridis",
    )


def add_target(viewer: _NapariViewer, scenario: "scenarios.Scenario3D") -> None:
    """Adds the target as a shapes layer to a napari Viewer.

    Args:
        viewer: The napari Viewer to which the target should be added.
        scenario: The 3D scenario which is being visualized.
    """
    target_pos = ((scenario.target_center - scenario.origin) / scenario.dx).astype(int)
    target_rad = int(scenario.target_radius / scenario.dx)
    dx, dy, dz = (
        np.array([target_rad, 0, 0]),
        np.array([0, target_rad, 0]),
        np.array([0, 0, target_rad]),
    )

    theta = np.pi / 4
    R1 = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    )
    theta = -np.pi / 4
    R2 = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    )

    ellipse_0 = np.array([-dx - dy, -dx + dy, dx + dy, dx - dy])
    ellipse_1 = target_pos + np.matmul(ellipse_0, R1)
    ellipse_2 = target_pos + np.matmul(ellipse_0, R2)
    ellipse_3 = np.array(
        [
            target_pos + (-dx - dz),
            target_pos + (-dx + dz),
            target_pos + (dx + dz),
            target_pos + (dx - dz),
        ]
    )

    viewer.add_shapes(
        name="target",
        data=[ellipse_1, ellipse_2, ellipse_3],
        shape_type="ellipse",
        edge_color="#2F2F2F",
        face_color="#999999",
        edge_width=1.0,
        opacity=0.5,
    )


def add_source(
    viewer: _NapariViewer, scenario: "scenarios.Scenario3D", source: "sources.Source"
) -> None:
    """Adds the source as a points layer to a napari Viewer.

    Each individual point source is plotted as a point.

    Args:
        viewer: The napari Viewer to which the target should be added.
        scenario: The 3D scenario which is being visualized.
        source: The source to be drawn.
    """
    source_coords = ((source.coordinates - scenario.origin) / scenario.dx).astype(int)
    viewer.add_points(
        name="sources",
        data=source_coords,
        symbol="o",
        size=1.0,
        face_color="#00AAFFFF",
        edge_color="#00AAFFFF",
        opacity=0.5,
    )
