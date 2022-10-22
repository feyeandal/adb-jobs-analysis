import os
import Levenshtein as lev
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd

import ocr_extraction

raw_image_path = "E:/ADB_Project/data/cs_sample_ocr"
transcript_path = "E:/ADB_Project/github/adb-jobs-analysis/data/ImageTransrciptions.csv"

def create_ocr_df(image_path):
    """Create a dataframe from a filepath"""

    # extracted_text_from_ocr
    ocr_extraction.extract_bulk(image_path)

    # make dataframe from extracte text
    ocr_df = pd.DataFrame(text, columns=["image_id", "file_path", "ocrd_text"])

    # change the datatype of the column
    ocr_df["vacancy_id"] = pd.to_numeric(ocr_df.vacancy_id, downcast='integer')

    return ocr_df

def read_transcript_file(file_path):
    """Read the transcripted text"""
    return pd.read_csv(file_path)

def merge_dataframes(df1, df2, column):
    """Merge two dataframes on two given columns"""
    return df1.merge(df2, on=column)

def calc_token_sort_ratio(token1, token2):
    """calcualates the token sort ration"""
    return fuzz.token_sort_ratio(token1, token2)

def display_results(series, eval_bins=[0,50,60,70,80,90,100])):
    """display the overall results"""
    print(df.column.value_counts(bins=eval_bins))

def main(raw_image_path, file_path, merge_column="image_id"):
    """Handles the direct evaluation of ocr output with hand transcribed text

    Parameters
    ----------
    parameter 1: str
        file location of the set of images used for the evaluation
    parameter 2: str
        file location of a csv containing hand_trasncribed text for the same set of images
    parameter 3: str
        name of the common column to merge ocr dataframe with the transcript dataframe
    """

    ocr_df = create_ocr_df(raw_image_path)

    transcript_df = read_transcript_file(file_path)

    merged_df =  merge_dataframes(ocr_df, transcript_df, on=merge_column)

    merged_df["token_sort_ratio"] = merged_df.apply(lambda x: calc_token_sort_ratio(x["ocrd_text"], x["transcribed_text"]), axis=1)

    display_results(merged_df.token_sort_ratio)




