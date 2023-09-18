import io
import base64
from neurotechdevkit.scenarios.built_in import BUILT_IN_SCENARIOS
from neurotechdevkit.scenarios import Scenario2D
from collections import namedtuple
from flask import (
    Flask,
    request,
    render_template,
)
import matplotlib

matplotlib.use("Agg")
from matplotlib import rc

rc("animation", html="html5")

app = Flask(__name__, static_folder="static", template_folder="templates")

Scenario = namedtuple("Scenario", ["id", "name", "is_2d"])


def _get_scenarios():
    scenarios = []
    for scenario_id, scenario in BUILT_IN_SCENARIOS.items():
        scenario = Scenario(id=scenario_id, name=scenario.__name__, is_2d=issubclass(scenario, Scenario2D))
        scenarios.append(scenario)
    return scenarios


@app.route("/")
async def index():
    title = "Neurotech Web App"
    return render_template("index.html", title=title, scenarios=_get_scenarios())


@app.route("/render_layout", methods=["POST"])
async def render_layout():
    print("received:", request.json)
    data = request.json
    # data = {
    #     "is_2d": True,
    #     "scenario": {
    #         "axis": "",
    #         "distance_from_origin": "",
    #         "is_prebuilt": False,
    #         "ct_file": "",
    #     },
    #     "transducer": {"transducers": []},
    #     "target": {"centerX": "0.0", "centerY": "0.0", "radius": "0.0"},
    #     "simulation_settings": {
    #         "simulation_precision": "5",
    #         "center_frequency": "500000",
    #         "material_properties": {
    #             "water": {
    #                 "vp": "1500",
    #                 "rho": "1000",
    #                 "alpha": "0.001100013975948576",
    #                 "render_color": "#2e86ab",
    #             },
    #             "trabecular_bone": {
    #                 "vp": "2300",
    #                 "rho": "1700",
    #                 "alpha": "1.7769765160626256",
    #                 "render_color": "#ebd378",
    #             },
    #             "brain": {
    #                 "vp": "1560",
    #                 "rho": "1040",
    #                 "alpha": "0.23999051039277436",
    #                 "render_color": "#db504a",
    #             },
    #         },
    #     },
    # }
    is_2d = data["is_2d"]
    scenario_params = data["scenario"]
    transducers = data["transducers"]
    target = data["target"]
    simulation_settings = data["simulation_settings"]

    if is_2d:
        if scenario_params.get("is_prebuilt", False):
            print('prebuilt scenario')
            scenario = BUILT_IN_SCENARIOS[scenario_params["scenario_id"]]()
            scenario.make_grid()
        else:
            print('custom scenario')
            scenario = Scenario2D()


        fig = scenario.render_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"<img src='data:image/png;base64,{data}'/>"
    return "Not supported"


if __name__ == "__main__":
    app.run(debug=True)
