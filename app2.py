# app.py
import os
import tempfile

from flask import Flask, render_template, request, jsonify

# Import from your existing script
# Make sure your big code is saved as analyzer.py in the same folder.
from analyzer import (
    predict_for_binary,
    load_model_metrics,
    Blockchain,
)

app = Flask(__name__)


def serialize_block(block):
    """Convert Block object to a JSON-serializable dict."""
    data = block.data if isinstance(block.data, dict) else {}
    return {
        "index": block.index,
        "timestamp": block.timestamp,
        "data": {
            "file": data.get("file"),
            "algo_id": data.get("algo_id"),
            "algo_name": data.get("algo_name"),
            "arch_name": data.get("arch_name"),
            "protocol_name": data.get("protocol_name"),
            "is_proprietary": data.get("is_proprietary"),
            "overall_metrics": data.get("overall_metrics"),
        },
        "previous_hash": block.previous_hash,
        "hash": block.hash,
    }


@app.route("/")
def index():
    # Renders templates/index.html
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Accept a binary file upload, run model predictions,
    update blockchain, and return JSON for the frontend.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save to temporary file for analysis
    tmp_fd, tmp_path = tempfile.mkstemp()
    os.close(tmp_fd)  # we will open via uploaded.save
    try:
        uploaded.save(tmp_path)

        # Run your existing ML pipeline
        prediction = predict_for_binary(tmp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Load metrics (if available)
    metrics_all = load_model_metrics()
    overall_metrics = (
        metrics_all.get("overall_ensemble") if isinstance(metrics_all, dict) else None
    )

    # Update blockchain
    chain = Blockchain()
    block_data = {
        "file": prediction.get("file"),
        "algo_id": prediction.get("algo_id"),
        "algo_name": prediction.get("algo_name"),
        "arch_name": prediction.get("arch_name"),
        "protocol_name": prediction.get("protocol_name"),
        "is_proprietary": prediction.get("ocsvm_is_proprietary"),
        "features": prediction.get("input_features"),
        "overall_metrics": overall_metrics,
    }
    new_block = chain.add_block(block_data)

    last_blocks = [serialize_block(b) for b in chain.chain[-5:]]

    return jsonify(
        {
            "prediction": prediction,
            "overall_metrics": overall_metrics,
            "blockchain": {
                "new_block": serialize_block(new_block),
                "last_blocks": last_blocks,
            },
        }
    )


if __name__ == "__main__":
    # For dev only; use gunicorn/uvicorn in production
    app.run(host="0.0.0.0", port=5000, debug=True)
