import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_ROOT = os.path.join(HERE, "tests")

OCR_ENGINES = ["easyocr", "tesseract"]
DIACRITIC_CHARS = set("ăâîșțĂÂÎȘȚ")


def has_diacritics(text: str) -> bool:
    return any(ch in DIACRITIC_CHARS for ch in text)


def check_engine(engine: str) -> bool:
    folder = os.path.join(TESTS_ROOT, engine)
    if not os.path.isdir(folder):
        print(f"[FAIL] Missing folder: {folder}")
        return False

    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(".txt")])
    if not files:
        print(f"[FAIL] No txt files in: {folder}")
        return False

    non_empty = 0
    with_diacritics = 0

    for filename in files:
        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if content:
            non_empty += 1
        if has_diacritics(content):
            with_diacritics += 1

    print(f"[OK] {engine}: total={len(files)}, non_empty={non_empty}, with_diacritics={with_diacritics}")

    if non_empty == 0:
        print(f"[FAIL] All files empty for {engine}")
        return False

    return True


def main():
    all_ok = True
    for engine in OCR_ENGINES:
        ok = check_engine(engine)
        all_ok = all_ok and ok

    if not all_ok:
        print("[RESULT] TESTS FAILED")
        sys.exit(1)

    print("[RESULT] TESTS PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
