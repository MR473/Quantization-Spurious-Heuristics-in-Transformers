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
parser.add_argument(
    "--seed", type=int, required=True, choices=[42, 123, 456]
)
args = parser.parse_args()

torch.set_num_threads(4)

seed = args.seed
model_path = Path(f"models/fp32/seed{seed}/final")
output_dir = Path("results/hans/weight_only_int8")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading seed {seed} from {model_path}", flush=True)

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

assert quantized_layers == 74, (
    f"Expected 74 quantized linear layers, got {quantized_layers}"
)

print(f"Quantized linear layers: {quantized_layers}", flush=True)

hans_url = (
    "https://raw.githubusercontent.com/tommccoy1/hans/"
    "master/heuristics_evaluation_set.txt"
)

hans = load_dataset(
    "csv",
    data_files={"validation": hans_url},
    delimiter="\t",
)["validation"]

columns_to_keep = [
    "gold_label",
    "heuristic",
    "subcase",
    "template",
    "pairID",
    "sentence1",
    "sentence2",
]

hans = hans.select_columns(columns_to_keep)
hans = hans.rename_columns({
    "sentence1": "premise",
    "sentence2": "hypothesis",
})

assert len(hans) == 30000
assert len(set(hans["pairID"])) == 30000

# Verify that the examples match our existing FP32 evaluation.
reference_path = Path(f"results/hans/fp32_seed{seed}_predictions.csv")
reference = pd.read_csv(reference_path)
reference_ids = reference["pairID"].astype(str).tolist()

assert hans["pairID"] == reference_ids, (
    "HANS example order differs from the original FP32 evaluation"
)

for column in [
    "gold_label", "heuristic", "subcase", "template",
    "premise", "hypothesis",
]:
    assert hans[column] == reference[column].tolist(), (
        f"HANS reference mismatch in {column}"
    )

print("HANS examples match existing FP32 predictions", flush=True)

def tokenize(batch):
    return tokenizer(
        batch["premise"],
        batch["hypothesis"],
        truncation=True,
        max_length=128,
    )

tokenized = hans.map(tokenize, batched=True)

inputs = tokenized.select_columns([
    "input_ids",
    "attention_mask",
    "token_type_ids",
])

loader = DataLoader(
    inputs,
    batch_size=32,
    shuffle=False,
    collate_fn=DataCollatorWithPadding(
        tokenizer=tokenizer,
        return_tensors="pt",
    ),
)

probabilities = []
predictions = []

with torch.inference_mode():
    for batch_index, batch in enumerate(loader):
        logits = model(**batch).logits
        probs = torch.softmax(logits, dim=-1)

        probabilities.extend(probs.cpu().tolist())
        predictions.extend(probs.argmax(dim=-1).cpu().tolist())

        if (batch_index + 1) % 100 == 0:
            print(
                f"Processed {len(predictions)}/{len(hans)}",
                flush=True,
            )

assert len(predictions) == len(hans)

results = hans.to_pandas()

results["mnli_prediction_id"] = predictions
results["mnli_prediction"] = [
    model.config.id2label[pred] for pred in predictions
]

results["hans_prediction"] = np.where(
    results["mnli_prediction_id"] == 0,
    "entailment",
    "non-entailment",
)

results["correct"] = (
    results["hans_prediction"] == results["gold_label"]
)

probs = np.asarray(probabilities)

assert probs.shape == (30000, 3)
assert np.isfinite(probs).all()

results["prob_entailment"] = probs[:, 0]
results["prob_neutral"] = probs[:, 1]
results["prob_contradiction"] = probs[:, 2]
results["prob_non_entailment"] = probs[:, 1] + probs[:, 2]

expected_columns = reference.columns.tolist()
assert set(results.columns) == set(expected_columns)
results = results[expected_columns]

overall_accuracy = float(results["correct"].mean())

lexical_non_entailment = results[
    (results["heuristic"] == "lexical_overlap")
    & (results["gold_label"] == "non-entailment")
]

lexical_accuracy = float(
    lexical_non_entailment["correct"].mean()
)

summary = {
    "model": "google-bert/bert-base-uncased",
    "seed": seed,
    "quantization": "int8_weight_only",
    "device": "cpu",
    "quantized_linear_layers": quantized_layers,
    "num_examples": len(results),
    "overall_accuracy": overall_accuracy,
    "lexical_non_entailment_accuracy": lexical_accuracy,
    "by_heuristic_and_label": {
        heuristic: {
            label: float(group["correct"].mean())
            for label, group in heuristic_df.groupby("gold_label")
        }
        for heuristic, heuristic_df in results.groupby("heuristic")
    },
}

prediction_path = (
    output_dir / f"weight_only_int8_seed{seed}_predictions.csv"
)
summary_path = (
    output_dir / f"weight_only_int8_seed{seed}_summary.json"
)

# Do not overwrite existing experimental results.
if prediction_path.exists() or summary_path.exists():
    raise FileExistsError(
        "Output already exists. Review it before rerunning."
    )

results.to_csv(prediction_path, index=False)
summary_path.write_text(json.dumps(summary, indent=2))

print("\n=== RESULTS ===")
print(f"Seed: {seed}")
print(f"HANS overall accuracy: {overall_accuracy:.4%}")
print(f"Lexical non-entailment accuracy: {lexical_accuracy:.4%}")
print(f"Saved predictions: {prediction_path}")
print(f"Saved summary: {summary_path}")
