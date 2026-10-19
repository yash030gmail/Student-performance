"""
Web Server and REST API for Student Performance Prediction Dashboard.
Serves:
  1. Interactive Web Dashboard UI at /
  2. Static Visualization Plots at /visualizations/<filename>
  3. Live ML Prediction REST endpoint at /api/predict (POST)
  4. Model Metrics and EDA Stats at /api/metrics and /api/stats
  5. Health Check at /health
"""

import os
import json
from flask import Flask, request, jsonify, send_file, send_from_directory
from predict import predict_student

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__, static_folder=None)

@app.route("/", methods=["GET"])
def index():
    dashboard_path = os.path.join(BASE_DIR, "dashboard.html")
    if os.path.exists(dashboard_path):
        return send_file(dashboard_path)
    return jsonify({"error": "dashboard.html not found"}), 404

@app.route("/visualizations/<path:filename>", methods=["GET"])
def get_visualization(filename):
    return send_from_directory(VIS_DIR, filename)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "service": "Student Performance Prediction Dashboard",
        "port": os.environ.get("PORT", 5050)
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    
    gender = data.get("gender", "female")
    race = data.get("race/ethnicity") or data.get("race", "group B")
    education = data.get("parental level of education") or data.get("education", "some college")
    lunch = data.get("lunch", "standard")
    prep = data.get("test preparation course") or data.get("prep", "completed")
    
    reading_score = data.get("reading score") or data.get("reading")
    writing_score = data.get("writing score") or data.get("writing")
    
    try:
        r_val = float(reading_score) if reading_score is not None and reading_score != "" else None
    except ValueError:
        r_val = None
        
    try:
        w_val = float(writing_score) if writing_score is not None and writing_score != "" else None
    except ValueError:
        w_val = None

    try:
        result = predict_student(
            gender=gender,
            race_ethnicity=race,
            parental_education=education,
            lunch=lunch,
            test_prep=prep,
            reading_score=r_val,
            writing_score=w_val
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Metrics not found"}), 404

@app.route("/api/stats", methods=["GET"])
def api_stats():
    stats_path = os.path.join(VIS_DIR, "eda_statistics.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "Stats not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"============================================================")
    print(f"  Student Performance Dashboard Server Running")
    print(f"  URL: http://localhost:{port}")
    print(f"  Health Check: http://localhost:{port}/health")
    print(f"============================================================")
    app.run(host="0.0.0.0", port=port, debug=False)
