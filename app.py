from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from utils.yolov8_inference import run_yolov8
import sqlite3
from datetime import datetime

app = Flask(__name__)
CORS(app)  # enable cross-origin requests for frontend JS
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DB_PATH = "database.db"

# ------------------ Database setup ------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    age INTEGER,
                    body_part TEXT,
                    notes TEXT,
                    xray_file TEXT,
                    annotated_file TEXT,
                    analysis_time TEXT
                )''')
    conn.commit()
    conn.close()

init_db()

# ------------------ Upload patient + X-ray ------------------
@app.route("/upload", methods=["POST"])
def upload():
    data = request.form
    file = request.files['xray']

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Save patient info + file to DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO patients(name, age, body_part, notes, xray_file, analysis_time)
                 VALUES(?,?,?,?,?,?)''',
              (data.get("name"), data.get("age"), data.get("body_part"),
               data.get("notes"), filename, datetime.now().isoformat()))
    patient_id = c.lastrowid
    conn.commit()
    conn.close()

    return jsonify({"success": True, "patient_id": patient_id, "filename": filename})

# ------------------ Run YOLO AI Analysis ------------------
@app.route("/analyze/<int:patient_id>", methods=["POST"])
def analyze(patient_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT xray_file FROM patients WHERE id=?", (patient_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({"error": "Patient not found"}), 404

    xray_file = row[0]
    xray_path = os.path.join(app.config['UPLOAD_FOLDER'], xray_file)

    # Run YOLO inference
    annotated_file, diagnosis = run_yolov8(xray_path, app.config['UPLOAD_FOLDER'])

    # Update DB with annotated image
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("UPDATE patients SET annotated_file=? WHERE id=?", (annotated_file, patient_id))
    conn.commit()
    conn.close()

    return jsonify({"annotated_file": annotated_file, "diagnosis": diagnosis})

# ------------------ Download image ------------------
@app.route("/download/<filename>", methods=["GET"])
def download(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)
