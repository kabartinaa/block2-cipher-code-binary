#!/usr/bin/env python3
import os, json, csv, random, binascii, zipfile

OUT_DIR = "crypto_hex_dataset_3000"
HEX_DIR = os.path.join(OUT_DIR, "hex_samples")
NUM_SAMPLES = 3000

ARCHS = ["x86", "x86_64", "arm", "arm64", "mips", "powerpc"]
PRIMITIVES = ["AES", "RSA", "ChaCha20", "SHA256", "SHA1", "GCM", "HMAC", "ECDHE", "ECDSA"]
PROTOCOLS = ["TLS1.2", "TLS1.3", "WPA2", "WPA3", "Bluetooth_LE", "Bluetooth_CLASSIC", "SSH", "IPSec", "DTLS"]

# Size distribution (bytes)
SIZE_BUCKETS = [(256, 0.25), (512, 0.25), (2048, 0.20), (4096, 0.20), (8192, 0.10)]

def pick_size():
    r = random.random()
    acc = 0
    for size, p in SIZE_BUCKETS:
        acc += p
        if r <= acc:
            return size
    return SIZE_BUCKETS[-1][0]

def rand_bytes(n):
    return bytes(random.getrandbits(8) for _ in range(n))

def inject_markers(raw, labels):
    # Optional: inject ASCII markers to aid parsing (kept sparse)
    marker = ("|ARCH:" + labels["architecture"] +
              "|PRIM:" + ",".join(labels["primitives"]) +
              "|PROTO:" + ",".join(labels["crypto_protocols"]) + "|").encode("ascii")
    pos = min(64, len(raw))
    return raw[:pos] + marker + raw[pos:]

def to_hex_tokens(raw):
    return " ".join(f"{b:02x}" for b in raw)

def main():
    os.makedirs(HEX_DIR, exist_ok=True)

    meta_csv_path = os.path.join(OUT_DIR, "metadata.csv")
    meta_jsonl_path = os.path.join(OUT_DIR, "metadata.jsonl")
    readme_path = os.path.join(OUT_DIR, "README.txt")

    csv_f = open(meta_csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.DictWriter(csv_f, fieldnames=[
        "sample_id","filename","architecture","primitives","crypto_protocols","length_bytes"
    ])
    csv_w.writeheader()
    jsonl_f = open(meta_jsonl_path, "w", encoding="utf-8")

    for i in range(1, NUM_SAMPLES+1):
        sample_id = f"s{i:04d}"
        arch = random.choice(ARCHS)
        # Pick 2–4 primitives, 1–3 protocols
        prims = sorted(random.sample(PRIMITIVES, k=random.randint(2,4)))
        prots = sorted(random.sample(PROTOCOLS, k=random.randint(1,3)))
        n = pick_size()

        raw = rand_bytes(n)
        raw = inject_markers(raw, {
            "architecture": arch,
            "primitives": prims,
            "crypto_protocols": prots
        })
        hex_str = to_hex_tokens(raw)

        fname = f"{sample_id}.hex"
        with open(os.path.join(HEX_DIR, fname), "w", encoding="utf-8") as hf:
            hf.write(hex_str)

        row = {
            "sample_id": sample_id,
            "filename": fname,
            "architecture": arch,
            "primitives": ",".join(prims),
            "crypto_protocols": ",".join(prots),
            "length_bytes": len(raw)
        }
        csv_w.writerow(row)
        jsonl_f.write(json.dumps(row) + "\n")

    csv_f.close()
    jsonl_f.close()

    with open(readme_path, "w", encoding="utf-8") as rf:
        rf.write(
            "Synthetic crypto-labeled hex dataset (3000 samples)\n\n"
            "Each .hex file contains space-separated byte tokens (00–ff) representing a binary-like blob.\n"
            "Labels include architecture, crypto primitives, and protocols in metadata files.\n"
            "Markers are injected near the start to help parsers.\n\n"
            "Files:\n"
            "- hex_samples/*.hex\n"
            "- metadata.csv\n"
            "- metadata.jsonl\n"
        )

    zip_path = "crypto_hex_dataset_3000.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(readme_path, arcname="README.txt")
        z.write(meta_csv_path, arcname="metadata.csv")
        z.write(meta_jsonl_path, arcname="metadata.jsonl")
        for root, _, files in os.walk(HEX_DIR):
            for f in files:
                full = os.path.join(root, f)
                rel = os.path.relpath(full, OUT_DIR)
                z.write(full, arcname=rel)

    print("Done. Created:", zip_path)

if __name__ == "__main__":
    main()
