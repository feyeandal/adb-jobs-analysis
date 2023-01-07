import os
import numpy as np
import pandas as pd

import logging
import enchant
import pytesseract
import image_preprocessing as ip
from text_preprocessing import strip_additional_characters, get_special_chars
 

def extract_text_one_image(image_path):
    """ extracts text for a single image"""
    return pytesseract.image_to_string(image_path)


 #change the declaration location 
def extract_text_bulk(folder_path, acc_threshold):
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

    low_accuracy_images = []
    cleaned_texts = []
    plain_accuracy = []
    clean_accuracy = []

    for i, vacancy in enumerate(all_images):
        logging.info(f'Processing {vacancy}: {i+1}/{num_images}')

        try:
            image_path = f"{folder_path}/{vacancy}"

            # Using pytesseract to extract text
            text = extract_text_one_image(image_path)
            stripped_text = strip_additional_characters(text)
            plain_accuracy_image = calculate_accuracy(text)
            clean_accuracy_image = calculate_accuracy(stripped_text)

            if plain_accuracy_image < acc_threshold:
                low_accuracy_images.append((image_path, vacancy))
            
            cleaned_texts.append(stripped_text)
            plain_accuracy.append(plain_accuracy_image)
            clean_accuracy.append(clean_accuracy_image)

            filepaths.append(image_path)
            vacancies.append(vacancy.split(".")[0])
            image2text.append(text)
            stripped_text = strip_additional_characters(text)
            cleaned_texts.append(strip_additional_characters(text))

        except: #TODO Catch a specific exception
            tesseract_failures.append(vacancy)
            logging.error(f"Tesseract Failure: {vacancy}")   
    
    logging.info(f'Out of {len(all_images)} images, {(len(low_accuracy_images)/len(all_images))* 100}% are below the accuracy threshold')

    ocrd = {
        "vacancy_id": vacancies,
        "file_path": filepaths,
        "ocrd_text": image2text,
        "clean_text": cleaned_texts,
        "plain_accuracy": plain_accuracy,
        "clean_accuracy": clean_accuracy
    }

    return ocrd, low_accuracy_images


def calculate_accuracy(string):
    """Checks a list of words against a dictionary and returns a ratio of valid words"""
    
    dic = enchant.Dict("en_US")
    
    valid_count = 0
    
    for word in string.split():
        if dic.check(word) == True:
            valid_count += 1
            
    return (valid_count/max(1, len(string.split()))) #TO DO - LATER


def update_ocr(low_accuracy_images, ocr_model_path, acc_threshold):
    """
    #for OCR accuracy values below a threshold, preprocess images to improve ocr and calculate accuracy metrics
    """
    logging.info("Preprocessing Images")

    preprocessing_failures = []

    read_images = [ip.read_image(img[0]) for img in low_accuracy_images]
    inverted_images = [ip.inversion(img) for img in read_images]
    binarized_images = [ip.grayscale(img) for img in inverted_images]
    upscaled_images = ip.super_res(binarized_images, ocr_model_path)
    eroded_images = [ip.thin_font(img) for img in upscaled_images]
    bordered_images = [ip.add_borders(img) for img in eroded_images]

    # logging.info(f'Thresholding accuracy at {acc_threshold}')

    # msk = df['clean_accuracy'] < acc_threshold
    # num_images_below_threshold = len(df[msk])

    # # ADD MULTIPLE COLUMNS THROUGH THE SAME LAMBDA FUNCTION WITHOUT LOOPING THROUGH SEVERAL TIMES

    # # TODO: This is currently inefficient given that the code is looping through the all the images several times
    # logging.info('Cleaning the images below the accuracy threshold..')
    
    ocrd_text = []
    cleaned_texts = []
    plain_accuracy = []
    clean_accuracy = [] 

    for img in bordered_images:
        text = extract_text_one_image(img)
        stripped_text = strip_additional_characters(text)
        plain_accuracy_image = calculate_accuracy(text)
        clean_accuracy_image = calculate_accuracy(stripped_text)

        ocrd_text.append(text)
        cleaned_texts.append(stripped_text)
        plain_accuracy.append(plain_accuracy_image)
        clean_accuracy.append(clean_accuracy_image)

    vacancy_id = [img[1] for img in low_accuracy_images]
    file_path = [img[0] for img in low_accuracy_images]
    
    df = pd.DataFrame([vacancy_id, file_path, ocrd_text, cleaned_texts, plain_accuracy, clean_accuracy]).transpose()
    df.columns = columns=["vacancy_id", "file_path", "ocrd_text", "clean_text", "plain_accuracy", "clean_accuracy"]


    msk = df['clean_accuracy'] < acc_threshold
    num_images_below_threshold_post_cleaning = len(df[msk])
    logging.info(f'Post cleaning, {(num_images_below_threshold_post_cleaning/len(df)) * 100}% images below threshold')

    return df


def main(read_path, save_path, ocr_model_path, acc_threshold):
    """ reads images from a directory and saves final ocr output to a csv"""
    #uses the ocr_extraction sub-module to conduct initial OCR and build a dataframe
    logging.info('Extracting text from images...')
    text, low_accuracy_images = extract_text_bulk(read_path, acc_threshold)
    ocr_df = pd.DataFrame.from_dict(text, orient="index" 
                        #   columns=["vacancy_id", "file_path", "ocrd_text", "clean_text", "plain_accuracy", "clean_accuracy"]
                        ).transpose() #insert column headers here
    
    #basic cleaning to strip additional characters 
    #logging.info('Cleaning the extracted text...')
    #ocr_df["clean_text"] = ocr_df["ocrd_text"].apply(strip_additional_characters)
    
    #evaluate ocr quality
    #logging.info('Evaluating accuracy of ocr quality')
    #ocr_df["plain_accuracy"] = ocr_df["ocrd_text"].apply(calculate_accuracy)
    #ocr_df["clean_accuracy"] = ocr_df["clean_text"].apply(calculate_accuracy)


    
    #iteratre through the dataset, identify poor quality ocr, preprocess images & perform ocr again
    ocr_df_cleaned = update_ocr(low_accuracy_images, ocr_model_path, acc_threshold)

    ocr_df = ocr_df.merge(ocr_df_cleaned)

    #save the final dataframe to a csv
    ocr_df.to_csv(save_path, index=False)
    
    return (ocr_df)
