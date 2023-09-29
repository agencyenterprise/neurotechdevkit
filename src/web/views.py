"""Views for the web app."""
import shutil
import tempfile
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request
from pydantic import ValidationError
from web.computed_tomography import validate_ct
from web.controller import (
    get_available_cts,
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


@bp.route("/")
async def index():
    """Render the index page, listing all the built-in scenarios."""
    return render_template(
        "index.html",
        title="Neurotech Web App",
        has_simulation=SimulationRunner().has_last_result,
        is_running_simulation=SimulationRunner().is_running,
        configuration=SimulationRunner().configuration,
        built_in_scenarios=get_built_in_scenarios(),
        all_materials=MaterialName.get_material_titles(),
        all_material_properties=get_default_material_properties(
            DEFAULT_CENTER_FREQUENCY
        ).dict(),
        all_transducer_types=TransducerType.get_transducer_titles(),
        available_cts=get_available_cts(current_app.config["CT_FOLDER"]),
    )


@bp.route("/simulate", methods=["POST"])
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
        data, image_format = result.data
        return f"<img src='data:image/{image_format};base64,{data}'/>"
    elif result.type == "no_simulation":
        return jsonify({"error": "No result"}), 400


@bp.route("/simulate", methods=["GET"])
def get_simulation():
    """Get the result of a finished or running simulation as a base64 GIF or PNG."""
    result = SimulationRunner().get()
    if result.type == "simulation":
        data, image_format = result.data
        return f"<img src='data:image/{image_format};base64,{data}'/>"
    elif result.type == "no_simulation":
        return jsonify({"error": "No result"}), 400


@bp.route("/simulate", methods=["DELETE"])
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
    image_format = "png"
    return f"<img src='data:image/{image_format};base64,{data}'/>"


@bp.route("/ct_scan", methods=["POST"])
def ct_scan():
    """Upload a Computed Tomography scan and return the available scans."""
    files = list(request.files.values())
    selected_filename = ""
    temp_dir = Path(tempfile.mkdtemp())
    saved_files = []
    for file in files:
        filename = secure_filename(file.filename)
        if not filename.endswith(".json"):
            selected_filename = filename
        file.save(temp_dir / filename)
        saved_files.append(filename)

    try:
        validate_ct(temp_dir, saved_files)
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
            "selected_ct": selected_filename,
        }
    )