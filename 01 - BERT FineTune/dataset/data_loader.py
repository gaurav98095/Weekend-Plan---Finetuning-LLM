import json
from datasets import Dataset, DatasetDict, Features, Sequence, ClassLabel, Value

# ==== Load and fix label list ====
with open("ner_labels.json") as f:
    raw_labels = json.load(f)
    if isinstance(raw_labels, dict):
        # Convert dict {"O": 0, "B-PER": 1, ...} to list by index
        ner_labels = [label for label, _ in sorted(raw_labels.items(), key=lambda x: x[1])]
    else:
        ner_labels = raw_labels

# ==== Dummy POS and Chunk tags ====
dummy_pos_labels = ["X"]
dummy_chunk_labels = ["X"]

# ==== Hugging Face Dataset Features ====
features = Features({
    "id": Value("string"),
    "tokens": Sequence(Value("string")),
    "pos_tags": Sequence(ClassLabel(names=dummy_pos_labels)),
    "chunk_tags": Sequence(ClassLabel(names=dummy_chunk_labels)),
    "ner_tags": Sequence(ClassLabel(names=ner_labels)),
})

# ==== Extract id2label and label2id from features ====
ner_class_label = features["ner_tags"].feature
id2label = {i: label for i, label in enumerate(ner_class_label.names)}
label2id = {label: i for i, label in enumerate(ner_class_label.names)}

# Optional: Save these mappings if needed
with open("id2label.json", "w") as f:
    json.dump(id2label, f, indent=2)

with open("label2id.json", "w") as f:
    json.dump(label2id, f, indent=2)

# ==== Load JSONL file and validate ====
def load_data(filename):
    data = []
    with open(filename) as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            tokens = obj["tokens"]
            tags = obj["tags"]
            if len(tokens) != len(tags):
                print(f"❌ Skipping line {i}: length mismatch ({len(tokens)} tokens vs {len(tags)} tags)")
                continue
            data.append({
                "id": str(i),
                "tokens": tokens,
                "pos_tags": [0] * len(tokens),   # All dummy "X"
                "chunk_tags": [0] * len(tokens), # All dummy "X"
                "ner_tags": tags
            })
    return data

# ==== Load datasets ====
dataset = DatasetDict({
    "train": Dataset.from_list(load_data("train.json")).cast(features),
    "validation": Dataset.from_list(load_data("valid.json")).cast(features),
    "test": Dataset.from_list(load_data("test.json")).cast(features)
})

# ==== Push to Hugging Face Hub ====
# Ensure you're logged in with `huggingface-cli login` or pass token below
dataset.push_to_hub("gaurav98095/ner-dataset")

# ==== Print mappings ====
print("id2label:", id2label)
print("label2id:", label2id)
