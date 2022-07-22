import os
import numpy as np
import pandas as pd
import enchant
import ocr_extraction
import image_preprocessing

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

path = "E:/ADB_Project/code/data/cs_sample"

# -------------------------------------------------------------------------------------------

def get_special_chars(text_column):
    
    """"identify special characters that need to be removed before evaluation"""
    
    #converting to a single string
    text = ' '.join(text_column)
    
    # get a list of unique characters
    text_char = list(set(text))
    
    # get a list removing alpha numeric
    text_char_sp = [char for char in text_char if not(char.isalnum())]
    
    return text_char_sp

def strip_special_chars(text, schar_list, char_keep):
    
    """Strips the unwanted special characters from a given list of job descriptions"""
    
    char_set = set([c for c in schar_list if c not in char_keep])
    
    # i2t_stripped -> stripped of special chars
    text_stripped = [''.join([c for c in item if c not in char_set]) for item in text]
    
    return text_stripped


def calculate_accuracy(string):
    """Checks a list of words against a dictionary and returns a ratio of valid words"""
    
    dic = enchant.Dict("en_US")
    
    valid_count = 0
    
    for word in string.split():
        if dic.check(word) == True:
            valid_count += 1
            
    return (valid_count/len(string.split()))

# ----------------------------------------------------------------------------------------------
def strip_additional_characters(ocr_list):
    """---"""
    # execute the function on the i2t list to get a list of special characters
    special = get_special_chars(ocr_list)

    # define characters you want to retain
    char_keep = [' ', '#', '+', '\n', '/']

    # execute the function and obtain ocr output stripped of special characters
    stripped = strip_special_chars(ocr_list, special, char_keep)

    return stripped

def compile_accuracy(df):
    """
    placeholder
    """
    df["plain_accuracy"] = df["ocrd_text"].apply(calculate_accuracy)
    df["clean_text"] = df.strip_additional_characters(df["ocrd_text"])
    df["clean_accuracy"] = df["clean_text"].apply(calculate_accuracy)
    return df

def update_ocr(df):
    """
    placeholder
    """
    df = compile_accuracy(df)

    #Loop through the dataframe, find bad OCR, preprocess images to improve ocr
    for index in df.index:
        if df.loc[index,'clean_accuracy'] < 0.9:
            vacancy = df.loc[index, 'image_id']
            binarized =  image_preprocessing.binarization(os.path.join(path, vacancy)) #update with extended functionality when read
            df.loc[index, 'ocrd_text'] = ocr_extraction.extract_text(binarized)
            df.loc[index, 'clean_text'] = df.strip_additional_characters(df["ocrd_text"])
            df.loc[index, 'plain_accuracy'] = calculate_accuracy(df.loc[index, 'ocrd_text'])
            df.loc[index, 'clean_accuracy'] = calculate_accuracy(df.loc[index, 'clean'])
    return df

    



