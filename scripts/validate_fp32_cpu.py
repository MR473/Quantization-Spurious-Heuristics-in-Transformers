import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results/hans/cpu_fp32_validation"
HANS_URL = (
    "https://raw.githubusercontent.com/tommccoy1/hans/"
    "master/heuristics_evaluation_set.txt"
)


def main(seed):
    torch.set_num_threads(4)
    device = torch.device("cpu")

    model_path = ROOT / f"models/fp32/seed{seed}/final"
    print(f"Loading FP32 checkpoint: {model_path}", flush=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True
    )

    assert model.config.label2id["entailment"] == 0
    model.to(device)
    model.eval()

    hans = load_dataset(
        "csv",
        data_files={"validation": HANS_URL},
        delimiter="\t",
    )["validation"]

    keep = [
        "gold_label", "heuristic", "subcase", "template",
        "pairID", "sentence1", "sentence2",
    ]
    hans = hans.select_columns(keep)
    hans = hans.rename_columns({
        "sentence1": "premise",
        "sentence2": "hypothesis",
    })

    assert len(hans) == 30000

    def tokenize(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=128,
        )

    tokenized = hans.map(tokenize, batched=True)
    inputs = tokenized.select_columns([
        "input_ids", "attention_mask", "token_type_ids"
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

    predictions = []
    probabilities = []

    with torch.inference_mode():
        for i, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)

            predictions.append(torch.argmax(probs, dim=-1).numpy())
            probabilities.append(probs.numpy())

            if (i + 1) % 100 == 0:
                print(f"Processed {(i + 1) * 32} examples", flush=True)

    predictions = np.concatenate(predictions)
    probabilities = np.concatenate(probabilities)

    df = hans.to_pandas()
    df["mnli_prediction_id"] = predictions
    df["mnli_prediction"] = [
        model.config.id2label[int(p)] for p in predictions
    ]
    df["hans_prediction"] = np.where(
        predictions == 0, "entailment", "non-entailment"
    )
    df["correct"] = df["hans_prediction"] == df["gold_label"]

    df["prob_entailment"] = probabilities[:, 0]
    df["prob_neutral"] = probabilities[:, 1]
    df["prob_contradiction"] = probabilities[:, 2]
    df["prob_non_entailment"] = (
        probabilities[:, 1] + probabilities[:, 2]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"fp32_cpu_seed{seed}_predictions.csv"
    df.to_csv(output, index=False)

    print(f"\nSaved: {output}")
    print(f"Examples: {len(df)}")
    print(f"Device: {device}")
    print(f"Overall accuracy: {df['correct'].mean() * 100:.4f}%")

    for label, group in df.groupby("gold_label"):
        print(f"{label}: {group['correct'].mean() * 100:.4f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[42, 123, 456], required=True)
    args = parser.parse_args()
    main(args.seed)
