import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    CLIPVisionModel,
    CLIPImageProcessor,
    AutoModel,
    AutoTokenizer,
)
from tqdm import tqdm


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

# Internal architecture constants — not exposed to pipeline writers.
# The multimodal model is fixed: CLIP-vision + mBERT + MLP head.
TEXT_MODEL = "bert-base-multilingual-cased"
VISION_MODEL = "openai/clip-vit-base-patch32"


@dataclass
class ExperimentConfig:
    """All settings needed to run a multimodal experiment.

    Pipeline writers build one of these per combination and call
    `run_experiment(cfg)` to train+evaluate.
    """
    tsv_file: str
    text_folder: str
    image_folder: str

    label_column: str = "Real_Fake"

    max_len: int = 128
    batch_size: int = 16
    epochs: int = 5
    lr: float = 1e-3
    seed: int = 42
    test_size: float = 0.2

    checkpoint_dir: Optional[str] = None
    resume: bool = True

    verbose: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(
    tsv_file: str,
    text_folder: str,
    image_folder: str,
    label_column: str,
) -> tuple[list[str], list[str], list]:

    df = pd.read_csv(tsv_file, sep="\t", dtype={"ID": str})

    if not os.path.exists(text_folder):
        raise FileNotFoundError(f"Text folder not found: {text_folder}")
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder not found: {image_folder}")

    available_text_files: dict[str, str] = {}
    for filename in os.listdir(text_folder):
        if filename.endswith(".txt"):
            file_id = filename.split("_")[0].replace(".txt", "")
            available_text_files[file_id] = filename

    available_image_files: dict[str, str] = {}
    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            file_id = os.path.splitext(filename)[0]
            available_image_files[file_id] = filename

    image_paths: list[str] = []
    texts: list[str] = []
    labels: list = []

    for _, row in df.iterrows():
        file_id = str(row["ID"])
        label = row[label_column]

        if pd.isna(label):
            continue
        if file_id not in available_text_files:
            continue
        if file_id not in available_image_files:
            continue

        text_path = os.path.join(text_folder, available_text_files[file_id])
        with open(text_path, "r", encoding="utf-8") as f:
            text_content = f.read().strip()
        if not text_content:
            continue

        image_paths.append(os.path.join(image_folder, available_image_files[file_id]))
        texts.append(text_content)
        labels.append(label)

    return image_paths, texts, labels


def encode_labels(
    labels: list,
    classes_path: Optional[str] = None,
) -> tuple[np.ndarray, LabelEncoder]:
    
    label_encoder = LabelEncoder()
    if classes_path is not None and os.path.exists(classes_path):
        label_encoder.classes_ = np.load(classes_path, allow_pickle=True)
        labels_encoded = label_encoder.transform(labels)
    else:
        labels_encoded = label_encoder.fit_transform(labels)
    return labels_encoded, label_encoder


class MemeMultimodalDataset(Dataset):
    def __init__(self, image_paths, texts, labels, image_processor, tokenizer, max_len):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        pixel_values = self.image_processor(
            images=img, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        enc = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes: int, vision_model_name: str, text_model_name: str):
        super().__init__()
        self.vision = CLIPVisionModel.from_pretrained(vision_model_name)
        self.text = AutoModel.from_pretrained(text_model_name)

        for p in self.vision.parameters():
            p.requires_grad = False
        for p in self.text.parameters():
            p.requires_grad = False

        vision_dim = self.vision.config.hidden_size
        text_dim = self.text.config.hidden_size

        self.head = nn.Sequential(
            nn.Linear(vision_dim + text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            v = self.vision(pixel_values=pixel_values).pooler_output
            t = self.text(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0, :]

        x = torch.cat([v, t], dim=1)
        return self.head(x)


def train_loop(
    model: MultimodalClassifier,
    train_loader: DataLoader,
    optimizer: AdamW,
    loss_function: CrossEntropyLoss,
    epochs: int,
    device: torch.device,
    verbose: bool = True,
) -> None:
    """Train model.head for `epochs` epochs."""
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, leave=True) if verbose else train_loader
        for batch in loop:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_b = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(pixel_values, input_ids, attention_mask)
            loss = loss_function(logits, labels_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if verbose:
                loop.set_description(f"Epoch {epoch + 1}/{epochs}")
                loop.set_postfix(loss=loss.item())

        if verbose:
            avg = total_loss / max(len(train_loader), 1)
            print(f"Epoch {epoch + 1} done. Average loss: {avg:.4f}")


def evaluate(
    model: MultimodalClassifier,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    predictions: list[int] = []
    gold: list[int] = []

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_b = batch["labels"].to(device)

            logits = model(pixel_values, input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy().tolist())
            gold.extend(labels_b.cpu().numpy().tolist())

    accuracy = accuracy_score(gold, predictions)
    macro_f1 = f1_score(gold, predictions, average="macro", zero_division=0)
    return accuracy, macro_f1, predictions, gold


def save_checkpoint(
    model: MultimodalClassifier,
    optimizer: AdamW,
    label_encoder: LabelEncoder,
    checkpoint_dir: str,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(model.head.state_dict(), os.path.join(checkpoint_dir, "mlp.pt"))
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    np.save(os.path.join(checkpoint_dir, "classes.npy"), label_encoder.classes_)


def run_experiment(cfg: ExperimentConfig) -> dict:
    
    set_seed(cfg.seed)
    device = get_device()

    # 1. load data
    if cfg.verbose:
        print("Loading data...")
    image_paths, texts, labels = load_data(
        cfg.tsv_file, cfg.text_folder, cfg.image_folder, cfg.label_column
    )
    if cfg.verbose:
        print(f"{len(texts)} valid (image + text) pairs loaded.")

    # 2. encode labels
    classes_path = (
        os.path.join(cfg.checkpoint_dir, "classes.npy")
        if cfg.checkpoint_dir and cfg.resume
        else None
    )
    labels_encoded, label_encoder = encode_labels(labels, classes_path=classes_path)
    num_classes = len(label_encoder.classes_)
    if cfg.verbose:
        print(f"Classes detected ({num_classes}): {list(label_encoder.classes_)}")

    # 3. split
    (
        train_image_paths,
        test_image_paths,
        train_texts,
        test_texts,
        train_labels,
        test_labels,
    ) = train_test_split(
        image_paths,
        texts,
        labels_encoded,
        test_size=cfg.test_size,
        random_state=cfg.seed,
        stratify=labels_encoded,
    )

    # 4. class weights
    class_weights_vals = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )
    class_weights = torch.tensor(class_weights_vals, dtype=torch.float).to(device)
    if cfg.verbose:
        print(f"Class weights: {class_weights_vals}")

    # 5. processors + datasets
    if cfg.verbose:
        print("Loading processors and tokenizer...")
    image_processor = CLIPImageProcessor.from_pretrained(VISION_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)

    train_dataset = MemeMultimodalDataset(
        train_image_paths, train_texts, train_labels,
        image_processor, tokenizer, cfg.max_len,
    )
    test_dataset = MemeMultimodalDataset(
        test_image_paths, test_texts, test_labels,
        image_processor, tokenizer, cfg.max_len,
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # 6. build model
    if cfg.verbose:
        print("Building model...")
    model = MultimodalClassifier(
        num_classes=num_classes,
        vision_model_name=VISION_MODEL,
        text_model_name=TEXT_MODEL,
    ).to(device)

    if cfg.checkpoint_dir and cfg.resume:
        mlp_path = os.path.join(cfg.checkpoint_dir, "mlp.pt")
        if os.path.exists(mlp_path):
            if cfg.verbose:
                print(f"[INFO] Resuming MLP head from {mlp_path}")
            model.head.load_state_dict(torch.load(mlp_path, map_location=device))

    # 7. optimizer + loss
    optimizer = AdamW(model.head.parameters(), lr=cfg.lr)
    loss_function = CrossEntropyLoss(weight=class_weights)

    if cfg.checkpoint_dir and cfg.resume:
        optimizer_path = os.path.join(cfg.checkpoint_dir, "optimizer.pt")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
            if cfg.verbose:
                print("Optimizer state loaded.")

    # 8. train
    if cfg.verbose:
        print("\nStarting training...")
    train_loop(
        model, train_loader, optimizer, loss_function,
        cfg.epochs, device, verbose=cfg.verbose,
    )

    # 9. save checkpoint (optional)
    if cfg.checkpoint_dir is not None:
        if cfg.verbose:
            print("\nSaving checkpoint...")
        save_checkpoint(model, optimizer, label_encoder, cfg.checkpoint_dir)
        if cfg.verbose:
            print(f"Saved to: {cfg.checkpoint_dir}")

    # 10. evaluate
    if cfg.verbose:
        print("\nEvaluating on test set...")
    accuracy, macro_f1, predictions, gold = evaluate(model, test_loader, device)

    if cfg.verbose:
        print(f"\nAccuracy: {accuracy * 100:.2f}%")
        print(f"Macro F1: {macro_f1:.4f}")
        print("\nClassification report:")
        print(
            classification_report(
                gold, predictions,
                target_names=list(label_encoder.classes_),
                digits=4, zero_division=0,
            )
        )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "n_total": len(texts),
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "classes": list(label_encoder.classes_),
        "predictions": predictions,
        "gold": gold,
    }


def append_results_md(
    record: dict,
    results_dir: str,
    base_name: str = "multimodal_metrics",
    dedup_keys: tuple[str, ...] = ("model", "engine", "stage"),
) -> tuple[str, str]:
    """Append a record to {base_name}.csv and rewrite {base_name}.md.

    Deduplicates by `dedup_keys` (keeps the most recent). Returns (csv_path, md_path).
    """
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{base_name}.csv")
    md_path = os.path.join(results_dir, f"{base_name}.md")

    new_df = pd.DataFrame([record])
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=list(dedup_keys), keep="last")
    else:
        combined = new_df

    combined = combined.sort_values(list(dedup_keys)).reset_index(drop=True)
    combined.to_csv(csv_path, index=False)

    display_df = combined.copy()
    if "accuracy" in display_df.columns:
        display_df["accuracy"] = display_df["accuracy"].map(lambda x: f"{x:.4f}")
    if "macro_f1" in display_df.columns:
        display_df["macro_f1"] = display_df["macro_f1"].map(lambda x: f"{x:.4f}")

    headers = list(display_df.columns)
    header_row = "| " + " | ".join(headers) + " |"
    sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    data_rows = [
        "| " + " | ".join(str(row[c]) for c in headers) + " |"
        for _, row in display_df.iterrows()
    ]
    md_table = "\n".join([header_row, sep_row] + data_rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table + "\n")

    return csv_path, md_path


if __name__ == "__main__":
    DEFAULT_CFG = ExperimentConfig(
        tsv_file=os.path.join(PROJECT_ROOT, "metadata.tsv"),
        text_folder=os.path.join(PROJECT_ROOT, "normalization", "tests", "easyocr"),
        image_folder=os.path.join(PROJECT_ROOT, "img"),
        label_column="Real_Fake",
        checkpoint_dir=os.path.join(HERE, "model_salvat_multimodal"),
        resume=True,
    )
    RESULTS_DIR = os.path.join(HERE, "comparison_results")
    ENGINE_TAG = "easyocr"
    STAGE_TAG = "normalization"

    result = run_experiment(DEFAULT_CFG)

    record = {
        "model": "multimodal",
        "engine": ENGINE_TAG,
        "stage": STAGE_TAG,
        "n_total": result["n_total"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "accuracy": round(result["accuracy"], 4),
        "macro_f1": round(result["macro_f1"], 4),
    }

    csv_path, md_path = append_results_md(
        record, RESULTS_DIR, base_name="multimodal_metrics"
    )

    print(f"\nResults saved to:\n  {csv_path}\n  {md_path}")
    with open(md_path, "r", encoding="utf-8") as f:
        print("\n" + f.read().rstrip())
