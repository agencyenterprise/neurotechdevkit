"""Views for the web app."""
from flask import Blueprint, jsonify, render_template, request
from pydantic import ValidationError

from web.controller import (
    get_built_in_scenarios,
    get_scenario_layout,
    get_simulation_image,
    get_supported_materials,
)
from web.messages import RenderLayoutRequest, SimulateRequest

bp = Blueprint("main", __name__, url_prefix="/")


@bp.route("/")
async def index():
    """Render the index page, listing all the built-in scenarios."""
    title = "Neurotech Web App"
    return render_template(
        "index.html",
        title=title,
        built_in_scenarios=get_built_in_scenarios(),
        materials=get_supported_materials(),
    )


@bp.route("/simulate", methods=["POST"])
async def simulate():
    """Simulate a scenario and return the result as a base64 gif or png."""
    try:
        config = SimulateRequest.parse_obj(request.json)
    except ValidationError as e:
        # If the JSON data doesn't match the Pydantic model,
        # return a 400 Bad Request response
        return jsonify({"error": str(e)}), 400

    data, image_format = get_simulation_image(config)
    return f"<img src='data:image/{image_format};base64,{data}'/>"


@bp.route("/render_layout", methods=["POST"])
async def render_layout():
    """Render the layout of a scenario and return the result as a base64 png."""
    try:
        config = RenderLayoutRequest.parse_obj(request.json)
    except ValidationError as e:
        # If the JSON data doesn't match the Pydantic model,
        # return a 400 Bad Request response
        return jsonify({"error": str(e)}), 400

    data = get_scenario_layout(config)
    image_format = "png"
    return f"<img src='data:image/{image_format};base64,{data}'/>"