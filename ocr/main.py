import os
import warnings
import gc

import cv2
import easyocr
import numpy as np
import pytesseract
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*pin_memory.*")

CLAHE_OBJ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def process_image_for_ocr(image_path, show_steps=False):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Alb negru
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Redimensionare 2x
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    enhanced_gray = CLAHE_OBJ.apply(gray)

    # Denoise
    final_img = cv2.medianBlur(enhanced_gray, 3)

    if show_steps:
        cv2.imshow("Imaginea finala pentru OCR", final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_img


def process_folder(folder_path):
    print("Se incarca modelul EasyOCR in memorie...")
    reader = easyocr.Reader(["ro"])
    
    # sa caute textul peste tot in imagine
    custom_config = r"--oem 3 --psm 11"

    if not os.path.exists(folder_path):
        print(f"Eroare: Folderul '{folder_path}' nu a fost gasit!")
        return

    os.makedirs("./tests/tesseract", exist_ok=True)
    os.makedirs("./tests/easyocr", exist_ok=True)

    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    print(f"S-au gasit {len(image_files)} imagini. Incep procesarea...")

    images_processed = 0

    for filename in tqdm(image_files, desc="Procesare imagini", unit="img"):
        image_path = os.path.join(folder_path, filename)
        base_name = os.path.splitext(filename)[0]

        tess_file_path = os.path.join("./tests/tesseract", f"{base_name}_tesseract.txt")
        easy_file_path = os.path.join("./tests/easyocr", f"{base_name}_easy_ocr.txt")

        # sarim peste procesare daca ambele fisiere exista deja
        if os.path.exists(tess_file_path) and os.path.exists(easy_file_path):
            continue

        processed_img = process_image_for_ocr(image_path)
        
        if processed_img is None:
            continue

        try:
            tesseract_text = pytesseract.image_to_string(
                processed_img, lang="ron", config=custom_config
            )

            with torch.no_grad():
                easyocr_results = reader.readtext(processed_img, detail=0)
            
            easyocr_text = "\n".join(easyocr_results)

            with open(tess_file_path, "w", encoding="utf-8") as f:
                f.write(tesseract_text.strip())

            with open(easy_file_path, "w", encoding="utf-8") as f:
                f.write(easyocr_text.strip())

        except Exception as e:
            with open("./tests/error_log.txt", "a", encoding="utf-8") as log:
                log.write(f"Eroare la {filename}: {str(e)}\n")

        finally:
            del processed_img
            if 'easyocr_results' in locals():
                del easyocr_results

            images_processed += 1

            if images_processed % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("\n" + "-" * 30 + "\nProcesarea s-a incheiat cu succes!")


if __name__ == "__main__":
    FOLDER_IMAGINI = "../img"
    process_folder(FOLDER_IMAGINI)