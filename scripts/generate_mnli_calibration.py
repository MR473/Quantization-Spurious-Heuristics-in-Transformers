import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchao.quantization import (
    quantize_,
    Int8DynamicActivationInt8WeightConfig,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, required=True, choices=[42, 123, 456])
parser.add_argument(
    "--split",
    choices=["validation_matched", "validation_mismatched"],
    default="validation_matched",
)
args = parser.parse_args()

torch.set_num_threads(4)

model_path = Path(f"models/fp32/seed{args.seed}/final")
output_dir = Path("results/calibration")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading seed {args.seed} from {model_path}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(str(model_path))
model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
model = model.cpu().eval()

quantize_(model, Int8DynamicActivationInt8WeightConfig())
model.eval()

print("INT8 quantization completed", flush=True)

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
            print(f"Processed {len(labels)}/{len(dataset)}", flush=True)

df = pd.DataFrame(probabilities, columns=[
    "prob_entailment",
    "prob_neutral",
    "prob_contradiction",
])
df.insert(0, "example_index", range(len(df)))
df["gold_label_id"] = labels
df["prediction_id"] = predictions
df["correct"] = df["gold_label_id"] == df["prediction_id"]

accuracy = float(df["correct"].mean())

original_results_path = Path(f"results/mnli/int8_seed{args.seed}_results.json")
original_results = json.loads(original_results_path.read_text())
metric_name = (
    "matched_accuracy"
    if args.split == "validation_matched"
    else "mismatched_accuracy"
)
original_accuracy = float(original_results[metric_name])

print(f"\nNew accuracy:      {accuracy:.8f}")
print(f"Original accuracy: {original_accuracy:.8f}")
print(f"Difference:        {accuracy - original_accuracy:+.8f}")

if abs(accuracy - original_accuracy) > 1e-6:
    raise RuntimeError(
        "Accuracy does not reproduce the original result. "
        "Calibration predictions were not saved."
    )

output_path = output_dir / f"int8_seed{args.seed}_{args.split}.csv"
df.to_csv(output_path, index=False)

print(f"Saved: {output_path}")
print(f"Rows: {len(df)}")
