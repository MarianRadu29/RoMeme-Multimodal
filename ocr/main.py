import os
import warnings

import cv2
import easyocr
import numpy as np
import pytesseract

warnings.filterwarnings("ignore", message=".*pin_memory.*")


def process_image_for_ocr(image_path, show_steps=True):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Alb negru
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Redimensionare 2x
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Imbunatatirea contrastului
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

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

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg")):
            image_path = os.path.join(folder_path, filename)

            base_name = os.path.splitext(filename)[0]

            print(f"Procesez: {filename} ...")

            processed_img = process_image_for_ocr(image_path)

            if processed_img is None:
                print(f"Eroare la citirea {filename}. Skip.")
                continue

            tesseract_text = pytesseract.image_to_string(
                processed_img, lang="ron", config=custom_config
            )

            easyocr_results = reader.readtext(processed_img, detail=0)
            easyocr_text = "\n".join(easyocr_results)

            tess_file_path = os.path.join(
                "./tests/tesseract", f"{base_name}_tesseract.txt"
            )
            with open(tess_file_path, "w", encoding="utf-8") as f:
                f.write(tesseract_text.strip())

            easy_file_path = os.path.join(
                "./tests/easyocr", f"{base_name}_easy_ocr.txt"
            )
            with open(easy_file_path, "w", encoding="utf-8") as f:
                f.write(easyocr_text.strip())

    print("-" * 30 + "\nProcesarea s-a incheiat cu succes!")


FOLDER_IMAGINI = "../img"

process_folder(FOLDER_IMAGINI)
