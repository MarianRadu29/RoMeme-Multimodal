import os
from tqdm import tqdm

from restorer import DiacriticsRestorer


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)

SOURCES = {
    "easyocr": os.path.join(PROJECT_ROOT, "normalization", "tests", "easyocr"),
    "tesseract": os.path.join(PROJECT_ROOT, "normalization", "tests", "tesseract"),
}

TARGETS = {
    "easyocr": os.path.join(HERE, "tests", "easyocr"),
    "tesseract": os.path.join(HERE, "tests", "tesseract"),
}


def diacritize_folder(src_dir: str, dst_dir: str, restorer: DiacriticsRestorer) -> int:
    if not os.path.isdir(src_dir):
        print(f"[skip] Source folder not found: {src_dir}")
        return 0

    os.makedirs(dst_dir, exist_ok=True)

    txt_files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(".txt")])
    processed = 0

    for filename in tqdm(txt_files, desc=f"Diacritizing {os.path.basename(src_dir)}", unit="file"):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        with open(src_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        restored_text = restorer.restore_diacritics(raw_text)

        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(restored_text)

        processed += 1

    return processed


def main():
    print("[diacritics] Starting batch diacritization for all OCR engines...")
    restorer = DiacriticsRestorer()

    total = 0
    for name in SOURCES:
        count = diacritize_folder(SOURCES[name], TARGETS[name], restorer)
        total += count
        print(f"[diacritics] {name}: {count} files -> {TARGETS[name]}")

    print(f"[diacritics] Done. Total processed files: {total}")

if __name__ == "__main__":
    main()

