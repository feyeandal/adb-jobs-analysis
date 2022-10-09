import numpy as np
import pandas as pd
import ocr_extraction
import ocr_evaluation
 
 #change the declaration location 

def main(read_path = "D:/nlp/top_jobs_cs_20_21/part_1/part_1b", save_path = "D:/nlp/top_jobs_cs_20_21/part_1/part_1b/p1b.csv"):
    """ reads images from a directory and saves final ocr output to a csv"""
    #uses the ocr_extraction sub-module to conduct initial OCR and build a dataframe
    text = ocr_extraction.extract_bulk(read_path)
    ocr_df = pd.DataFrame(text, columns=["vacancy_id", "file_path", "ocrd_text"]) #insert column headers here
    
    #basic cleaning to strip additional characters 
    ocr_df["clean_text"] = ocr_df["ocrd_text"].apply(ocr_evaluation.strip_additional_characters)
    
    #evaluate ocr quality
    ocr_df["plain_accuracy"] = ocr_df["ocrd_text"].apply(ocr_evaluation.calculate_accuracy)
    ocr_df["clean_accuracy"] = ocr_df["clean_text"].apply(ocr_evaluation.calculate_accuracy)
    
    #TO DO - Add logging to the pipeline to: 1. mark checkpoints, and 2. see how it's performing

    #iteratre through the dataset, identify poor quality ocr, preprocess images & perform ocr again
    ocr_df = ocr_evaluation.update_ocr(ocr_df)

    #save the final dataframe to a csv
    ocr_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()
