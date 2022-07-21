import os
import numpy as np
import pandas as pd
import pytesseract
import enchant

import ocr_extraction
import preprocess_images

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


def accuracy_calculator(string):
    """Checks a list of words against a dictionary and returns a ratio of valid words"""
    
    dic = enchant.Dict("en_US")
    
    valid_count = 0
    
    for word in string.split():
        if dic.check(word) == True:
            valid_count += 1
            
    return (valid_count/len(string.split()))

# ----------------------------------------------------------------------------------------------

text = ocr.extract_text(path, n=10)
df = pd.DataFrame(text, index=np.arange(10))

i2t = list(df["ocrd_text"])

# execute the function on the i2t list to get a list of special characters
special = get_special_chars(i2t)

# define characters you want to retain
char_keep = [' ', '#', '+', '\n', '/']

# execute the function and obtain ocr output stripped of special characters
stripped = strip_special_chars(i2t, special, char_keep)

df["clean"] = pd.Series(stripped)
#accuracy calculation
df["plain_accuracy"] = df["ocrd_text"].apply(accuracy_calculator)
df["clean_accuracy"] = df["clean"].apply(accuracy_calculator)


for index in df.index:
    if df.loc[index,'clean_accuracy'] < 0.9:
        vacancy = df.loc[index, 'job_id']
        binarized =  preprocess_images.binarization(os.path.join(path, vacancy))
        df.loc[index, 'ocrd_text'] = pytesseract.image_to_string(binarized)
        special = get_special_chars([df.loc[index, 'ocrd_text']])
        df.loc[index, 'clean'] = strip_special_chars([df.loc[index, 'ocrd_text']], special, char_keep)[0]
        df.loc[index, 'plain_accuracy'] = accuracy_calculator(df.loc[index, 'ocrd_text'])
        df.loc[index, 'clean_accuracy'] = accuracy_calculator(df.loc[index, 'clean'])
        
# ----------------------------------------------------------------------------------------------

save_path = "E:/ADB_Project/code/data/pipeline_sample.csv"

df.to_csv(save_path, index=False)




