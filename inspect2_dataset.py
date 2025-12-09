import os, collections, math, csv

FOLDER = "crypto_hex_dataset_3000/hex_samples"
OUTFILE = "hex_features.csv"

def read_hex_file(path):
    with open(path) as f:
        tokens = f.read().split()
    return bytes(int(t,16) for t in tokens)

def entropy(data):
    counts = collections.Counter(data)
    total = len(data)
    return -sum((c/total) * math.log2(c/total) for c in counts.values())

def detect_markers(raw_bytes):
    try:
        text = raw_bytes.decode("latin1", errors="ignore")
        if "|ARCH:" in text:
            start = text.find("|ARCH:")
            end = text.find("|", start+1)
            return text[start:end+1]
    except:
        pass
    return ""

def main():
    with open(OUTFILE, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "filename","length","entropy","mean","variance","min","max","marker"
        ])
        writer.writeheader()

        for fname in sorted(os.listdir(FOLDER)):
            if not fname.endswith(".hex"): continue
            raw = read_hex_file(os.path.join(FOLDER, fname))
            vals = list(raw)
            row = {
                "filename": fname,
                "length": len(raw),
                "entropy": entropy(raw),
                "mean": sum(vals)/len(vals),
                "variance": sum((x - sum(vals)/len(vals))**2 for x in vals)/len(vals),
                "min": min(vals),
                "max": max(vals),
                "marker": detect_markers(raw)
            }
            writer.writerow(row)

    print("Done. Features saved to", OUTFILE)

if __name__ == "__main__":
    main()
