# t2_inspect_labels.py
import json
from collections import Counter

TASK2_TRAIN = "../task2/task2_train.jsonl"
TASK2_DEV = "../task2/task2_dev.jsonl"

def read_labels(path):
    label_counter = Counter()
    label_texts = {}
    with open(path, "r") as f:
        for line in f:
            obj = json.loads(line)
            lab = obj["label"]
            lab_text = obj.get("label_text", "")
            label_counter[lab] += 1
            if lab not in label_texts and lab_text:
                label_texts[lab] = lab_text
    return label_counter, label_texts

def main():
    for split, path in [("train", TASK2_TRAIN), ("dev", TASK2_DEV)]:
        print(f"=== {split} ===")
        cnt, texts = read_labels(path)
        print("Label distribution:", cnt)
        print("Label_text examples:")
        for lab, txt in sorted(texts.items()):
            print(f"  label {lab}: {txt}")
        print()

if __name__ == "__main__":
    main()
