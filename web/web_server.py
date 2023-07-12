from flask import Flask, render_template, send_file, make_response, url_for, Response, request
import mpld3
import pathlib
import tempfile
from werkzeug.utils import secure_filename


from neurotechdevkit.api import create_scenario
import io
import base64
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)


@app.route('/')
async def index():
    title = "Neurotech Web App"
    return render_template("index.html", title=title)

def handle_file_upload(request):
    ct_file = None
    if 'ct_file' in request.files:
        f = request.files['ct_file']
        path = secure_filename(f.filename)
        f.save(path)
        ct_file = pathlib.Path(path)
    return ct_file

@app.route("/plot", methods=["POST"])
async def plot():
    jsdata = request.form.to_dict()
    jsdata['ct_file'] = handle_file_upload(request)

    scenario = create_scenario(jsdata)
    fig = scenario.render_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

@app.route("/run_simulation", methods=["POST"])
async def run_simulation():
    jsdata = request.form.to_dict()
    jsdata['ct_file'] = handle_file_upload(request)
    scenario = create_scenario(jsdata)
    result = scenario.simulate_steady_state()
    fig = result.render_steady_state_amplitudes()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"

@app.route("/plot_mpld3", methods=["POST"])
def plot_mpld3():
    jsdata = request.form.to_dict()
    jsdata['ct_file'] = handle_file_upload(request)
    scenario = create_scenario(jsdata)
    fig = scenario.render_layout()

    html_str = mpld3.fig_to_html(fig)
    return Response(html_str, mimetype='text/html')

if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmpdirname:
        app.config['UPLOAD_FOLDER'] = tmpdirname
        app.run(debug = True)