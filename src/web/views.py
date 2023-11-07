"""Views for the web app."""
import base64
import io
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from flask import Blueprint, current_app, jsonify, render_template, request
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.ticker import FuncFormatter
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
    elif result.type == "simulation_error":
        return jsonify({"error": str(result.error)}), 400
    elif result.type == "no_simulation":
        return jsonify({"error": "No result"}), 400


@bp.route("/simulate", methods=["GET"])
def get_simulation():
    """Get the result of a finished or running simulation as a base64 GIF or PNG."""
    result = SimulationRunner().get()
    if result.type == "simulation":
        data, image_format = result.data
        return f"<img src='data:image/{image_format};base64,{data}'/>"
    elif result.type == "simulation_error":
        return jsonify({"error": str(result.error)}), 400
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

    fig = get_scenario_layout(config)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    plt.close(fig)
    image_format = "png"
    return f"<img src='data:image/{image_format};base64,{data}'/>"


@bp.route("/render_canvas", methods=["POST"])
async def render_canvas():
    """Render the canvas of a scenario and return the result as a base64 PNG."""
    try:
        config = RenderLayoutRequest.parse_obj(request.json)
    except ValidationError as e:
        # If the JSON data doesn't match the Pydantic model,
        # return a 400 Bad Request response
        return jsonify({"error": str(e)}), 400

    fig = get_scenario_layout(config)
    plot_params = cleanup_plot(fig)
    return jsonify(plot_params)


def cleanup_plot(fig) -> dict:
    """Clean up the plot and return the parameters to render it in the front end."""
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    ax = fig.get_axes()[0]
    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Lists to store tick locations
    xticks = []
    yticks = []

    # Define function to capture x and y ticks when they are being formatted
    def capture_xticks(value, pos):
        xticks.append(value)
        return f"{value:.0f}"

    def capture_yticks(value, pos):
        yticks.append(value)
        return f"{value:.0f}"

    # Set FuncFormatters that use the above functions
    ax.xaxis.set_major_formatter(FuncFormatter(capture_xticks))
    ax.yaxis.set_major_formatter(FuncFormatter(capture_yticks))

    # Remove the title
    fig.suptitle("")

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    data_ratio = abs((xlim[1] - xlim[0]) / (ylim[1] - ylim[0]))
    fig.set_size_inches(4 * data_ratio, 4)  # 4 inches is 400 pixels

    canvas = FigureCanvas(fig)
    png_output = io.BytesIO()
    canvas.print_png(png_output)
    return {
        "image": base64.b64encode(png_output.getvalue()).decode("utf-8"),
        "xlim": xlim,
        "ylim": ylim,
        "xticks": xticks,
        "yticks": yticks,
        "xlabel": xlabel,
        "ylabel": ylabel,
    }


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
