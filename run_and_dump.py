import json
from integration3 import train_all_models, predict_for_binary  # change name

BIN_PATH = r"D:\P_2_S_2.bin"   # or any binary path you want
OUT_JSON = "prediction.json"

# Make sure models exist (same logic as in main())
needed = [
    "gnn_model.pth",
    "lstm_opcode_model.pth",
    "xgb_algo_model.json",
    "xgb_arch_model.json",
    "xgb_proto_model.json",
    "ocsvm_model.pkl",
    "algo_feature_means.json",
]
if not all(__import__("os").path.exists(f) for f in needed):
    print("[*] Some model files are missing. Training all models now...")
    train_all_models()
else:
    print("[+] All model files found. Skipping training.")

res = predict_for_binary(BIN_PATH)
with open(OUT_JSON, "w") as f:
    json.dump(res, f, indent=2)

print(f"[+] Saved prediction to {OUT_JSON}")
