# app.py
from flask import Flask, request, jsonify
from integration3 import predict_for_binary  # import the function

app = Flask(__name__)

@app.route("/api/predict", methods=["GET"])
def api_predict():
    bin_path = request.args.get("file")
    if not bin_path:
        return jsonify({"error": "file parameter required"}), 400
    res = predict_for_binary(bin_path)
    return jsonify(res)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


