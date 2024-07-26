"""Flask app factory."""

import os
import pathlib
import shutil
import webbrowser
import zipfile

import matplotlib
from flask import Flask
from flask_cors import CORS
from matplotlib import rc
from web.simulation_runner import SimulationRunner

CT_FOLDER = pathlib.Path("./ct-scans/")
WEB_SERVER_HOST = os.getenv("WEB_SERVER_HOST", "localhost")
WEB_SERVER_PORT = os.getenv("WEB_SERVER_PORT", 8080)
CURRENT_DIR = pathlib.Path(__file__).parent
APP_DIR = CURRENT_DIR / "app"


def create_app():
    """Create the Flask app."""
    matplotlib.use("Agg")

    rc("animation", html="html5")

    app = Flask(__name__, static_folder="app/dist")
    CORS(app)

    # Register blueprints here
    from web.views import bp

    app.register_blueprint(bp)

    SimulationRunner().initialize()

    return app


def assign_CT_FOLDER(app):
    """Assign the CT_FOLDER to the app."""
    if not os.path.exists(CT_FOLDER):
        os.makedirs(CT_FOLDER)
    app.config["CT_FOLDER"] = CT_FOLDER


def extract_static_files():
    """
    Extract Vue build from dist.zip to app folder, so that it can be served by aiohttp
    """
    dist_dir = APP_DIR / "dist"
    if os.path.exists(dist_dir) and os.path.isdir(dist_dir):
        shutil.rmtree(dist_dir)
    with zipfile.ZipFile(APP_DIR / "dist.zip", "r") as zip_ref:
        zip_ref.extractall(APP_DIR / "dist")


app = create_app()
extract_static_files()


def run():
    """Run the web app."""
    assign_CT_FOLDER(app)
    webbrowser.open(f"http://{WEB_SERVER_HOST}:{WEB_SERVER_PORT}")
    app.run(host=WEB_SERVER_HOST, port=WEB_SERVER_PORT)


if __name__ == "__main__":
    run()
