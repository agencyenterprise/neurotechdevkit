"""Flask app factory."""
import os
import pathlib

import matplotlib
from flask import Flask
from matplotlib import rc
from web.simulation_runner import SimulationRunner

CT_FOLDER = pathlib.Path("./ct-scans/")


def create_app():
    """Create the Flask app."""
    matplotlib.use("Agg")

    rc("animation", html="html5")

    app = Flask(__name__, static_folder="static", template_folder="templates")

    # Register blueprints here
    from web.views import bp

    app.register_blueprint(bp)

    SimulationRunner().initialize()

    return app


def assign_CT_FOLDER(app):
    if not os.path.exists(CT_FOLDER):
        os.makedirs(CT_FOLDER)
    app.config["CT_FOLDER"] = CT_FOLDER


app = create_app()


def run():
    """Run the web app."""
    assign_CT_FOLDER(app)
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    run()
