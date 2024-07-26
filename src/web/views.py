"""Views for the web app."""

import os
import shutil
import tempfile
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_from_directory
from pydantic import ValidationError
from web.computed_tomography import get_available_cts, validate_ct
from web.controller import (
    get_built_in_scenarios,
    get_default_material_properties,
    get_scenario_layout,
    get_simulation_image,
)
from web.messages.material_properties import MaterialName
from web.messages.requests import RenderLayoutRequest, SimulateRequest
from web.messages.transducers import TransducerType
from web.simulation_runner import SimulationRunner
from werkzeug.utils import secure_filename

bp = Blueprint("main", __name__, url_prefix="/")

DEFAULT_CENTER_FREQUENCY = 5e5


@bp.route("/", defaults={"path": ""})
@bp.route("/<path:path>")
def serve_spa(path):
    """Serve the single page app."""
    if path != "" and os.path.exists(os.path.join(current_app.static_folder, path)):
        return send_from_directory(current_app.static_folder, path)
    else:
        return send_from_directory(current_app.static_folder, "index.html")


@bp.route("/info")
async def info():
    """Render the initial page, listing all the built-in scenarios."""
    return jsonify(
        has_simulation=SimulationRunner().has_last_result,
        is_running_simulation=SimulationRunner().is_running,
        configuration=SimulationRunner().configuration,
        built_in_scenarios=get_built_in_scenarios(),
        materials=MaterialName.get_material_titles(),
        material_properties=get_default_material_properties(
            DEFAULT_CENTER_FREQUENCY
        ).dict(),
        transducer_types=TransducerType.get_transducer_titles(),
        available_cts=get_available_cts(current_app.config["CT_FOLDER"]),
    )


@bp.route("/simulation", methods=["POST"])
def simulate():
    """Simulate a scenario and return the result as a base64 GIF or PNG."""
    try:
        config = SimulateRequest.parse_obj(request.json)
    except ValidationError as e:
        # If the JSON data doesn't match the Pydantic model,
        # return a 400 Bad Request response
        return jsonify({"error": str(e)}), 400
    result = SimulationRunner().run(
        get_simulation_image(config=config, app_config=current_app.config),
        config.dict(),
    )
    if result.type == "simulation":
        data, _ = result.data
        return jsonify(
            {
                "data": data,
            }
        )
    elif result.type == "simulation_error":
        return jsonify({"error": str(result.error)}), 400
    elif result.type == "no_simulation":
        return jsonify({"error": "No result"}), 400


@bp.route("/simulation", methods=["GET"])
def get_simulation():
    """Get the result of a finished or running simulation as a base64 GIF or PNG."""
    result = SimulationRunner().get()
    if result.type == "simulation":
        data, _ = result.data
        return jsonify(
            {
                "data": data,
            }
        )
    elif result.type == "simulation_error":
        return jsonify({"error": str(result.error)}), 400
    elif result.type == "no_simulation":
        return jsonify({"error": "No result"}), 400


@bp.route("/simulation", methods=["DELETE"])
def remove_simulation():
    """Remove the result of a finished or running simulation."""
    SimulationRunner().reset()
    return "removed"


@bp.route("/render_layout", methods=["POST"])
async def render_layout():
    """Render the layout of a scenario and return the result as a base64 PNG."""
    try:
        config = RenderLayoutRequest.parse_obj(request.json)
    except ValidationError as e:
        # If the JSON data doesn't match the Pydantic model,
        # return a 400 Bad Request response
        return jsonify({"error": str(e)}), 400

    data = get_scenario_layout(config)
    return jsonify({"data": data})


@bp.route("/ct_scan", methods=["POST"])
def ct_scan():
    """Upload a Computed Tomography scan and return the available scans."""
    files = list(request.files.values())
    temp_dir = Path(tempfile.mkdtemp())
    saved_files = []
    for file in files:
        filename = secure_filename(file.filename)
        file.save(temp_dir / filename)
        saved_files.append(filename)

    try:
        ct_info = validate_ct(temp_dir, saved_files)
        for filename in saved_files:
            # moving files from the temporary directory to the CT_FOLDER
            (temp_dir / filename).rename(current_app.config["CT_FOLDER"] / filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        shutil.rmtree(temp_dir)

    return jsonify(
        {
            "available_cts": get_available_cts(current_app.config["CT_FOLDER"]),
            "selected_ct": {
                "filename": ct_info.filename,
                "shape": ct_info.shape,
                "spacing": ct_info.spacing,
            },
        }
    )
