# normalize_all.py
# Reads raw OCR outputs from ../ocr/tests/{easyocr,tesseract}/
# and writes normalized text to ./tests/{easyocr,tesseract}/ (inside this folder)

import os

from normalization import TextNormalizer

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
OCR_TESTS = os.path.join(PROJECT_ROOT, "ocr", "tests")
OUTPUT_DIR = os.path.join(HERE, "tests")

SOURCES = {
    "easyocr": os.path.join(OCR_TESTS, "easyocr"),
    "tesseract": os.path.join(OCR_TESTS, "tesseract"),
}
TARGETS = {
    "easyocr": os.path.join(OUTPUT_DIR, "easyocr"),
    "tesseract": os.path.join(OUTPUT_DIR, "tesseract"),
}


def normalize_folder(normalizer: TextNormalizer, src_dir: str, dst_dir: str):
    if not os.path.isdir(src_dir):
        print(f"[skip] source folder not found: {src_dir}")
        return

    os.makedirs(dst_dir, exist_ok=True)

    processed = 0
    for filename in sorted(os.listdir(src_dir)):
        if not filename.lower().endswith(".txt"):
            continue

        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        with open(src_path, encoding="utf-8") as f:
            raw = f.read()

        normalized = normalizer.normalize(raw)

        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(normalized)

        processed += 1

    print(f"[done] {src_dir} -> {dst_dir} ({processed} files)")


def main():
    print("--- Phase 3: normalizing all OCR outputs ---")
    normalizer = TextNormalizer()

    for name in SOURCES:
        normalize_folder(normalizer, SOURCES[name], TARGETS[name])

    print("--- Done. Normalized files ready for Phase 4 (diacritics). ---")


if __name__ == "__main__":
    main()
