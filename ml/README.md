# NLP Classifier for OCR-Extracted Texts

This project contains a Machine Learning script (based on PyTorch and Hugging Face Transformers) that classifies texts extracted from images (via EasyOCR) into categories such as `Real`, `Fake`, or `DeepFake`. The base model used is `bert-base-multilingual-cased`.

## 📌 Main Features

- **Fast Evaluation (Inference Mode):** By setting the number of epochs to `0`, the script loads an already saved model and directly runs the evaluation on the test data, without spending time on training.
- **Checkpoint Support:** Automatically saves and loads the state of the model, optimizer, and classes. If a saved model exists in `model_salvat_ocr/`, training or evaluation resumes automatically.
- **Class Imbalance Handling:** Automatically calculates and applies `class_weights` in the loss function (`CrossEntropyLoss`) so that majority categories aren't favored during training.
- **Flexible File Mapping:** Automatically finds `.txt` files extracted with EasyOCR and maps them to the corresponding IDs in the `metadata.tsv` table.

## 📂 File Structure

For the script to run correctly, the directory structure must match the paths defined in the configuration variables:

```text
RoMeme-Multimodal/
│
├── metadata.tsv                        <-- The file containing labels
├── normalization/
│   └── tests/
│       └── easyocr/                    <-- Folder with extracted texts (e.g., 00100001_easyocr.txt)
│
└── ml/
    ├── main.py                         <-- The main script
    └── model_salvat_ocr/               <-- Folder containing the trained model
        ├── config.json
        ├── model.safetensors
        ├── optimizer.pt
        └── classes.npy                 <-- Class mapping (e.g., 0=DeepFake, 1=Fake)
```

## 🚀 How to Use

### 1. Install dependencies

Make sure you have activated your environment (e.g., `.venv`) and installed the necessary packages:

```bash
pip install torch torchvision torchaudio transformers pandas numpy scikit-learn tqdm
```

### 2. Evaluation Mode (Current)

Currently, the script has `EPOCHS = 0`. This is the ideal setup when you already have a saved model and just want to test its performance on the 20% test data.

- The script will load the weights directly from the `./model_salvat_ocr` folder.
- It will skip the training loop entirely.
- It will pass the data through the model and output the `Total Accuracy`.

Run:

```bash
python main.py
```

### 3. Training Mode

If you want to train the model further or add new data:

1. Change the `EPOCHS = 0` variable in `main.py` to a higher value (e.g., `EPOCHS = 3`).
2. The script will load the current state of the model and continue learning, saving the new progress at the end of the run.
3. If you want to train the model entirely from scratch, simply delete or rename the `model_salvat_ocr` folder.

## 📊 Modifiable Configurations (Settings)

You can find these variables at the top of the `main.py` script:

- `TSV_FILE`: Path to the metadata table.
- `TEXT_FOLDER`: Path to the folder with the OCR text results.
- `LABEL_COLUMN`: Target column for classification (currently set to `Real_Fake`).
- `CHECKPOINT_DIR`: The directory where the model is saved/loaded from.
