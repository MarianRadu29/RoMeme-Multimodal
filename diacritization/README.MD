## Diacritization Module

This folder contains scripts for restoring Romanian diacritics on OCR text.

### Structure

- `restorer.py`: Loads the Hugging Face model and exposes `DiacriticsRestorer`.
- `diacritize_all.py`: Batch script that reads OCR `.txt` files and writes diacritized outputs.
- `test_diacritization.py`: Validation script for generated outputs per OCR engine.

### Input / Output

- Input folders:
  - `ocr/tests/easyocr`
  - `ocr/tests/tesseract`
- Output folders:
  - `diacritization/tests/easyocr`
  - `diacritization/tests/tesseract`

The script keeps the same filenames, so IDs stay aligned with `metadata.tsv`.

### Dependencies

Install required packages:

```bash
pip install transformers torch sentencepiece tqdm protobuf tiktoken
```

### Run

From project root:

```bash
python diacritization/diacritize_all.py
```

### Run Tests

After generating outputs, run:

```bash
python diacritization/test_diacritization.py
```

The test checks both OCR outputs (`easyocr` and `tesseract`) and validates:

- output folders exist
- there are `.txt` files
- files are not all empty
- how many files contain Romanian diacritic characters

### Notes

- The model is downloaded on first run and may take a while.
- If CUDA is available, inference runs on GPU automatically.
- Text is split into token-safe chunks to avoid truncation on long files.
