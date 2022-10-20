# https://www.youtube.com/watch?v=tQGgGY8mTP0&list=PL2VXyKi-KpYuTAZz__9KVl1jQz74bDG7i

import pandas as pd
import time
import pytesseract
import os
from pathlib import Path

def extract_text(image_path):
    """ extracts text for a single image"""
    return pytesseract.image_to_string(image_path)

def extract_bulk(path):
    """extract texts from all images in a directory and returns a dictionary of image id, filepath and ocr output"""
    filepaths = []
    vacancies = []
    image2text = []
    tesseract_failures = []
    
    for vacancy in os.listdir(path):
        try:
            filepaths.append(f"{path}/{vacancy}")
            vacancies.append(vacancy.split(".")[0])
            text = extract_text(f"{path}/{vacancy}")
            image2text.append(text)
        except:
            tesseract_failures.append(vacancy)
            print(f"Tesseract Failure: {vacancy}")    
        
    ocrd = {
        "vacancy_id": vacancies,
        "file_path": filepaths,
        "ocrd_text": image2text
        
    }

    return ocrd