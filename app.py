from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from inference import run_inference
from diagnosis import generate_diagnosis

UPLOAD_FOLDER = "uploads/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "dcm"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file selected"
        file = request.files["file"]

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Run anomaly detection
            results = run_inference(filepath)

            # Generate preliminary diagnosis
            diagnosis = generate_diagnosis(results)

            return render_template(
                "result.html",
                image_path=filepath,
                annotated_image=results["annotated_path"],
                diagnosis=diagnosis,
            )

    return render_template("upload.html")

if __name__ == "__main__":
    app.run(debug=True)
