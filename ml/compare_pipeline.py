import argparse
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

STAGE_DIRS = {
    "ocr": os.path.join(PROJECT_ROOT, "ocr", "tests"),
    "normalization": os.path.join(PROJECT_ROOT, "normalization", "tests"),
    "diacritization": os.path.join(PROJECT_ROOT, "diacritization", "tests"),
}
STAGES = ["ocr", "normalization", "diacritization"]
ENGINES = ["easyocr", "tesseract"]


class OCRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


@dataclass
class ExperimentConfig:
    model_name: str
    max_len: int
    batch_size: int
    epochs: int
    lr: float
    test_size: float
    seed: int
    label_column: str


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_id(filename: str) -> str:
    stem = os.path.splitext(filename)[0]
    return stem.split("_")[0]


def load_texts_for_folder(folder_path: str):
    texts_by_id = {}
    if not os.path.isdir(folder_path):
        return texts_by_id

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(".txt"):
            continue

        file_id = extract_id(filename)
        path = os.path.join(folder_path, filename)

        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if content:
            texts_by_id[file_id] = content

    return texts_by_id


def train_and_evaluate(
    train_texts,
    train_labels,
    test_texts,
    test_labels,
    num_labels,
    cfg: ExperimentConfig,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=num_labels,
    ).to(device)

    train_dataset = OCRDataset(train_texts, train_labels, tokenizer, cfg.max_len)
    test_dataset = OCRDataset(test_texts, test_labels, tokenizer, cfg.max_len)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    class_weights_vals = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )
    class_weights = torch.tensor(class_weights_vals, dtype=torch.float).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    loss_function = CrossEntropyLoss(weight=class_weights)

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False)
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_function(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"    - Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

    model.eval()
    predictions = []
    gold = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            gold.extend(labels.cpu().numpy())

    accuracy = accuracy_score(gold, predictions)
    macro_f1 = f1_score(gold, predictions, average="macro")
    return accuracy, macro_f1


def run_comparison(tsv_path: str, cfg: ExperimentConfig):
    print(f"Loading metadata from: {tsv_path}")
    metadata = pd.read_csv(tsv_path, sep="\t", dtype={"ID": str})

    id_to_label = {
        str(row["ID"]): row[cfg.label_column]
        for _, row in metadata.iterrows()
        if pd.notna(row[cfg.label_column])
    }

    records = []

    for engine in ENGINES:
        print("\n" + "=" * 70)
        print(f"Engine: {engine}")

        stage_texts = {}
        for stage in STAGES:
            folder = os.path.join(STAGE_DIRS[stage], engine)
            texts_by_id = load_texts_for_folder(folder)
            stage_texts[stage] = texts_by_id
            print(f"  {stage:<14} -> {len(texts_by_id)} non-empty files")

        common_ids = set.intersection(*[set(stage_texts[s].keys()) for s in STAGES])
        common_ids = sorted([fid for fid in common_ids if fid in id_to_label])

        if len(common_ids) < 10:
            print("  [skip] Not enough common IDs for fair comparison.")
            continue

        labels_raw = [id_to_label[fid] for fid in common_ids]
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels_raw)

        train_ids, test_ids, train_labels, test_labels = train_test_split(
            common_ids,
            labels_encoded,
            test_size=cfg.test_size,
            random_state=cfg.seed,
            stratify=labels_encoded,
        )

        id_to_encoded_label = {
            fid: int(lbl)
            for fid, lbl in zip(common_ids, labels_encoded)
        }

        for stage in STAGES:
            print(f"\n  Running stage: {stage}")
            texts_map = stage_texts[stage]

            stage_train_texts = [texts_map[fid] for fid in train_ids]
            stage_test_texts = [texts_map[fid] for fid in test_ids]
            stage_train_labels = [id_to_encoded_label[fid] for fid in train_ids]
            stage_test_labels = [id_to_encoded_label[fid] for fid in test_ids]

            acc, macro_f1 = train_and_evaluate(
                stage_train_texts,
                stage_train_labels,
                stage_test_texts,
                stage_test_labels,
                num_labels=len(label_encoder.classes_),
                cfg=cfg,
            )

            print(f"    accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")

            records.append(
                {
                    "engine": engine,
                    "stage": stage,
                    "n_total": len(common_ids),
                    "n_train": len(train_ids),
                    "n_test": len(test_ids),
                    "accuracy": acc,
                    "macro_f1": macro_f1,
                }
            )

    return pd.DataFrame.from_records(records)


def format_markdown_table(df: pd.DataFrame) -> str:
    display_df = df.copy()
    display_df["accuracy"] = display_df["accuracy"].map(lambda x: f"{x:.4f}")
    display_df["macro_f1"] = display_df["macro_f1"].map(lambda x: f"{x:.4f}")

    headers = list(display_df.columns)
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"

    data_rows = []
    for _, row in display_df.iterrows():
        values = [str(row[col]) for col in headers]
        data_rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header_row, separator_row] + data_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare OCR vs normalization vs diacritization for text classification."
    )
    parser.add_argument("--tsv", default=os.path.join(PROJECT_ROOT, "metadata.tsv"))
    parser.add_argument("--label-column", default="Real_Fake")
    parser.add_argument("--model-name", default="bert-base-multilingual-cased")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(HERE, "comparison_results"),
        help="Where to save CSV/Markdown results.",
    )

    args = parser.parse_args()

    cfg = ExperimentConfig(
        model_name=args.model_name,
        max_len=args.max_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        test_size=args.test_size,
        seed=args.seed,
        label_column=args.label_column,
    )

    set_seed(cfg.seed)

    results_df = run_comparison(args.tsv, cfg)
    if results_df.empty:
        print("No results were produced.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "comparison_metrics.csv")
    md_path = os.path.join(args.output_dir, "comparison_metrics.md")

    results_df = results_df.sort_values(["engine", "stage"]).reset_index(drop=True)
    results_df.to_csv(csv_path, index=False)

    md_table = format_markdown_table(results_df)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table + "\n")

    print("\n" + "=" * 70)
    print("FINAL COMPARATIVE TABLE")
    print(md_table)
    print("=" * 70)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
