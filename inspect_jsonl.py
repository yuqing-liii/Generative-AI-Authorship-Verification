import json

def read_first_n(path, n=3):
    print(f"\n=== Reading file: {path} ===")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            print(f"\n--- sample {i} ---")
            print("keys:", list(obj.keys()))  # Fields contained in each sample

            # Print each field
            for k, v in obj.items():
                if isinstance(v, str) and len(v) > 100:
                    print(f"{k}: {v[:100]}...")  # Truncate long text
                else:
                    print(f"{k}: {v}")
            if i + 1 >= n:
                break

if __name__ == "__main__":
    # Read train.jsonl and val.jsonl by default
    read_first_n("train.jsonl", n=3)
    read_first_n("val.jsonl", n=3)
