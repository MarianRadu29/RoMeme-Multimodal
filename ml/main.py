import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# ================= SETTINGS =================
TSV_FILE = '../metadata.tsv'           
TEXT_FOLDER = '../normalization/tests/easyocr/'     
LABEL_COLUMN = 'Real_Fake'          
MODEL_NAME = 'bert-base-multilingual-cased' 
CHECKPOINT_DIR = './model_salvat_ocr' 

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. LOADING DATA =================
print("Loading data...")
df = pd.read_csv(TSV_FILE, sep='\t', dtype={'ID': str})

texts = []
labels = []
ids_found = []

available_files = {}
if os.path.exists(TEXT_FOLDER):
    for filename in os.listdir(TEXT_FOLDER):
        if filename.endswith('.txt'):
            file_id_from_name = filename.split('_')[0].replace('.txt', '')
            available_files[file_id_from_name] = filename
else:
    print(f"Error: Folder {TEXT_FOLDER} does not exist!")

for index, row in df.iterrows():
    file_id = str(row['ID'])
    label = row[LABEL_COLUMN]
    
    if file_id in available_files:
        actual_filename = available_files[file_id]
        file_path = os.path.join(TEXT_FOLDER, actual_filename)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read().strip()
            if len(text_content) > 0:
                texts.append(text_content)
                labels.append(label)
                ids_found.append(file_id)

print(f"{len(texts)} valid text files found and loaded.")

label_encoder = LabelEncoder()
classes_path = os.path.join(CHECKPOINT_DIR, "classes.npy")

if os.path.exists(classes_path):
    label_encoder.classes_ = np.load(classes_path)
    labels_encoded = label_encoder.transform(labels)
else:
    labels_encoded = label_encoder.fit_transform(labels)

num_categories = len(label_encoder.classes_)
print(f"Categories detected ({num_categories}): {label_encoder.classes_}")

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# --- Calculate Class Weights ---
class_weights_vals = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
class_weights = torch.tensor(class_weights_vals, dtype=torch.float).to(device)
print(f"Class weights applied to correct imbalance: {class_weights_vals}")

# ================= 2. DEFINING DATASET =================
class OCRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer(
            text, add_special_tokens=True, max_length=self.max_len,
            padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- Loading Intelligent Model (Resuming vs New) ---
if os.path.exists(CHECKPOINT_DIR) and os.path.exists(os.path.join(CHECKPOINT_DIR, "config.json")):
    print("\n[INFO] A model saved was found! Resuming training...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_DIR, num_labels=num_categories)
else:
    print("\n[INFO] No saved model found. Starting a new training...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_categories)

model = model.to(device)

train_dataset = OCRDataset(train_texts, train_labels, tokenizer, MAX_LEN)
test_dataset = OCRDataset(test_texts, test_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================= 3. TRAINING THE MODEL =================
optimizer = AdamW(model.parameters(), lr=2e-5)
loss_function = CrossEntropyLoss(weight=class_weights) 

optimizer_path = os.path.join(CHECKPOINT_DIR, "optimizer.pt")
if os.path.exists(optimizer_path):
    optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))
    print("Optimizer state loaded.")

print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        loss = loss_function(outputs.logits, labels)
        
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        loop.set_description(f"Epoca {epoch+1}/{EPOCHS}")
        loop.set_postfix(loss=loss.item())
        
    print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(train_loader):.4f}")

# ================= 4. SAVING THE MODEL (CHECKPOINT) =================
print("\nSaving the model progress...")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
model.save_pretrained(CHECKPOINT_DIR)
tokenizer.save_pretrained(CHECKPOINT_DIR)
torch.save(optimizer.state_dict(), os.path.join(CHECKPOINT_DIR, "optimizer.pt"))
np.save(classes_path, label_encoder.classes_)
print(f"Save successful in folder: {CHECKPOINT_DIR}")

# ================= 5. EVALUATION =================
print("\nStarting evaluation on test data...")
model.eval()
predictions = []
real_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        
        predictions.extend(preds.cpu().numpy())
        real_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(real_labels, predictions)
print(f"\nTotal Accuracy: {accuracy * 100:.2f}%")
