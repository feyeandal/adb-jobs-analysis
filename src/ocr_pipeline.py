import os
import numpy as np
import pandas as pd

import logging
import enchant
import pytesseract
import image_preprocessing
from text_preprocessing import strip_additional_characters, get_special_chars
 

def extract_text_one_image(image_path):
    """ extracts text for a single image"""
    return pytesseract.image_to_string(image_path)


 #change the declaration location 
def extract_text_bulk(folder_path):
    """ Extracts text from all images in a given directory
    args:
        folder_path (str): absolute path for the folder
    
    return:
        ocrd [Dict]: A dictionary containing lists of image_ids, file_paths, and the ocr_output
    """

    filepaths = []
    vacancies = []
    image2text = []
    tesseract_failures = []

    all_images = os.listdir(folder_path)
    num_images = len(all_images)
    logging.info(f'Folder contains {num_images} images')

    for i, vacancy in enumerate(all_images):
        logging.info(f'Processing {vacancy}: {i+1}/{num_images}')

        try:
            image_path = f"{folder_path}/{vacancy}"

            # Using pytesseract to extract text
            text = extract_text_one_image(image_path)

            filepaths.append(image_path)
            vacancies.append(vacancy.split(".")[0])
            image2text.append(text)

        except: #TODO Catch a specific exception
            tesseract_failures.append(vacancy)
            logging.error(f"Tesseract Failure: {vacancy}")   

    ocrd = {
        "vacancy_id": vacancies,
        "file_path": filepaths,
        "ocrd_text": image2text
    }

    return ocrd


def calculate_accuracy(string):
    """Checks a list of words against a dictionary and returns a ratio of valid words"""
    
    dic = enchant.Dict("en_US")
    
    valid_count = 0
    
    for word in string.split():
        if dic.check(word) == True:
            valid_count += 1
            
    return (valid_count/max(1, len(string.split()))) #TO DO - LATER


def update_ocr(df, ocr_model_path, threshold=0.85):
    """
    #for OCR accuracy values below a threshold, preprocess images to improve ocr and calculate accuracy metrics
    """

    logging.info(f'Thresholding accuracy at {threshold}')

    msk = df['clean_accuracy'] < threshold
    num_images_below_threshold = len(df[msk])

    logging.info(f'Out of {len(df)} images, {(num_images_below_threshold/len(df))* 100}% are below the accuracy threshold')

    # TODO: This is currently inefficient given that the code is looping through the all the images several times
    logging.info('Cleaning the images below the accuracy threshold..')
    df['ocrd_text'] = df.apply(lambda x: extract_text_one_image(image_preprocessing.main(x['file_path'], ocr_model_path)) if (x['clean_accuracy'] < threshold) else x['ocrd_text'], axis=1)
   
    df['clean_text'] = df.apply(lambda x: strip_additional_characters(x['ocrd_text']) if (x['clean_accuracy'] < threshold) else x['clean_text'], axis=1)
    
    df['plain_accuracy'] = df.apply(lambda x: calculate_accuracy(x['ocrd_text']) if (x['clean_accuracy'] < threshold) else x['plain_accuracy'], axis=1)
    
    df['clean_accuracy'] = df.apply(lambda x: calculate_accuracy(x['clean_text']) if (x['clean_accuracy'] < threshold) else x['clean_accuracy'], axis=1)

    msk = df['clean_accuracy'] < threshold
    num_images_below_threshold_post_cleaning = len(df[msk])
    logging.info(f'Post cleaning, {(num_images_below_threshold_post_cleaning/len(df)) * 100}% images below threshold')

    return df


def main(read_path, save_path, ocr_model_path):
    """ reads images from a directory and saves final ocr output to a csv"""
    #uses the ocr_extraction sub-module to conduct initial OCR and build a dataframe
    logging.info('Extracting text from images...')
    text = extract_text_bulk(read_path)
    ocr_df = pd.DataFrame(text, columns=["vacancy_id", "file_path", "ocrd_text"]) #insert column headers here
    
    #basic cleaning to strip additional characters 
    logging.info('Cleaning the extracted text...')
    ocr_df["clean_text"] = ocr_df["ocrd_text"].apply(strip_additional_characters)
    
    #evaluate ocr quality
    logging.info('Evaluating accuracy of ocr quality')
    ocr_df["plain_accuracy"] = ocr_df["ocrd_text"].apply(calculate_accuracy)
    ocr_df["clean_accuracy"] = ocr_df["clean_text"].apply(calculate_accuracy)
    
    #iteratre through the dataset, identify poor quality ocr, preprocess images & perform ocr again
    ocr_df = update_ocr(ocr_df, ocr_model_path)

    #save the final dataframe to a csv
    ocr_df.to_csv(save_path, index=False)
    
    return (ocr_df)
