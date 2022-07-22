# https://www.youtube.com/watch?v=tQGgGY8mTP0&list=PL2VXyKi-KpYuTAZz__9KVl1jQz74bDG7i

import pandas as pd
import time
import pytesseract
import os
from pathlib import Path

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

def extract_text(image_path):
    return pytesseract.image_to_string(image_path)

def extract_bulk(path, n=3):
    """ returns a list containing OCR'd text from a given directory"""
    
    indexes = []
    image2text = []
    
    for vacancy in os.listdir(path)[:n]:
        indexes.append(vacancy)
        text = extract_text(f"{path}/{vacancy}")
        image2text.append(text)
        
    ocrd = {
        "image_id": indexes,
        "ocrd_text": image2text
    }

    return ocrd

