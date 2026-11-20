import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchao.quantization import quantize_, Int8WeightOnlyConfig
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, choices=[42, 123, 456])
parser.add_argument(
    "--split",
    required=True,
    choices=["validation_matched", "validation_mismatched"],
)
args = parser.parse_args()

torch.set_num_threads(4)

model_path = Path(f"models/fp32/seed{args.seed}/final")
output_dir = Path("results/mnli/weight_only_int8")

prediction_path = output_dir / (
    f"weight_only_int8_seed{args.seed}_{args.split}.csv"
)
summary_path = output_dir / (
    f"weight_only_int8_seed{args.seed}_{args.split}_summary.json"
)

if prediction_path.exists() or summary_path.exists():
    raise FileExistsError("Output already exists. Refusing to overwrite.")

print(f"Loading seed {args.seed} from {model_path}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    str(model_path), local_files_only=True
)
model = AutoModelForSequenceClassification.from_pretrained(
    str(model_path), local_files_only=True
).cpu().eval()

assert model.config.id2label == {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

quantize_(model, Int8WeightOnlyConfig())
model.eval()

quantized_layers = sum(
    type(module.weight).__name__ == "Int8Tensor"
    for module in model.modules()
    if isinstance(module, torch.nn.Linear)
)
assert quantized_layers == 74

print(f"Quantized linear layers: {quantized_layers}", flush=True)

dataset = load_dataset("nyu-mll/glue", "mnli")[args.split]

def tokenize(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=128,
    )

tokenized = dataset.map(tokenize, batched=True)

columns = ["input_ids", "attention_mask", "label"]
if "token_type_ids" in tokenized.column_names:
    columns.insert(1, "token_type_ids")

inputs = tokenized.select_columns(columns)

loader = DataLoader(
    inputs,
    batch_size=32,
    shuffle=False,
    collate_fn=DataCollatorWithPadding(tokenizer),
)

probabilities = []
predictions = []
labels = []

with torch.inference_mode():
    for batch_index, batch in enumerate(loader):
        batch_labels = batch.pop("labels")

        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)

        probabilities.extend(probs.cpu().tolist())
        predictions.extend(probs.argmax(dim=-1).cpu().tolist())
        labels.extend(batch_labels.cpu().tolist())

        if (batch_index + 1) % 50 == 0:
            print(
                f"Processed {len(labels)}/{len(dataset)}",
                flush=True,
            )

assert len(labels) == len(dataset)
assert len(predictions) == len(dataset)

probs = np.asarray(probabilities)
assert probs.shape == (len(dataset), 3)
assert np.isfinite(probs).all()
assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

df = pd.DataFrame(
    probs,
    columns=[
        "prob_entailment",
        "prob_neutral",
        "prob_contradiction",
    ],
)

df.insert(0, "example_index", range(len(df)))
df["gold_label_id"] = labels
df["prediction_id"] = predictions
df["correct"] = df["gold_label_id"] == df["prediction_id"]

accuracy = float(df["correct"].mean())

summary = {
    "model": "google-bert/bert-base-uncased",
    "seed": args.seed,
    "quantization": "int8_weight_only",
    "device": "cpu",
    "split": args.split,
    "num_examples": len(dataset),
    "quantized_linear_layers": quantized_layers,
    "accuracy": accuracy,
}

output_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(prediction_path, index=False)
summary_path.write_text(json.dumps(summary, indent=2))

print("\n=== RESULTS ===")
print(f"Seed: {args.seed}")
print(f"Split: {args.split}")
print(f"Accuracy: {accuracy:.4%}")
print(f"Saved predictions: {prediction_path}")
print(f"Saved summary: {summary_path}")
