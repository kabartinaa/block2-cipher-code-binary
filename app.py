from flask import Flask, request, jsonify, send_from_directory
import tempfile
import os
from integration3 import predict_for_binary

app = Flask(__name__, static_folder="static")

# -----------------------------------------
# 1. Serve FRONTEND
# -----------------------------------------
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

# Allow serving JS/CSS
@app.route("/<path:path>")
def serve_static_files(path):
    return send_from_directory(app.static_folder, path)

# -----------------------------------------
# 2. API for File Upload
# -----------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]

    if uploaded_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        uploaded_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # Run model
        result = predict_for_binary(tmp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return jsonify(result)

# -----------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)