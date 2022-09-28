import os
import numpy as np
import pandas as pd
import enchant
import ocr_extraction
import image_preprocessing

PATH = "E:/ADB_Project/code/data/cs_sample"

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
    text_stripped = ' '.join(''.join([c for c in item if c not in char_set]) for item in text.split())
    
    return text_stripped


def calculate_accuracy(string):
    """Checks a list of words against a dictionary and returns a ratio of valid words"""
    
    dic = enchant.Dict("en_US")
    
    valid_count = 0
    
    for word in string.split():
        if dic.check(word) == True:
            valid_count += 1
            
    return (valid_count/max(1, len(string.split()))) #TO DO - LATER

# ----------------------------------------------------------------------------------------------
def strip_additional_characters(ocr_list): #TO DO  - CONSOLIDATE FUNCTION
    """handles the stripping of non alpha-numeric characters for a list of strings"""
    # execute the function on the i2t list to get a list of special characters
    special = get_special_chars(ocr_list)

    # define characters you want to retain
    char_keep = [' ', '#', '+', '\n', '/']

    # execute the function and obtain ocr output stripped of special characters
    stripped = strip_special_chars(ocr_list, special, char_keep)

    return stripped

def update_ocr(df, threshold=0.85):
    """
    #for OCR accuracy values below a threshold, preprocess images to improve ocr and calculate accuracy metrics
    """
    
    df['ocrd_text'] = df.apply(lambda x: ocr_extraction.extract_text(image_preprocessing.binarization(x['file_path'])) if (x['clean_accuracy'] < threshold) else x['ocrd_text'], axis=1)
    df['clean_text'] = df.apply(lambda x: strip_additional_characters(x['ocrd_text']) if (x['clean_accuracy'] < threshold) else x['clean_text'], axis=1)
    df['plain_accuracy'] = df.apply(lambda x: calculate_accuracy(x['ocrd_text']) if (x['clean_accuracy'] < threshold) else x['plain_accuracy'], axis=1)
    df['clean_accuracy'] = df.apply(lambda x: calculate_accuracy(x['clean_text']) if (x['clean_accuracy'] < threshold) else x['clean_accuracy'], axis=1)

    return df

    



