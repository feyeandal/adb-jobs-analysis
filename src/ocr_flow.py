import numpy as np
import pandas as pd
import ocr_extraction
import ocr_evaluation
 
read_path = "E:/ADB_Project/code/data/cs_sample"
save_path = "E:/ADB_Project/code/data/pipeline_sample2.csv" #change the declaration location 

def main():
    """ reads images from a directory and saves final ocr output to a csv"""
    #uses the ocr_extraction sub-module to conduct initial OCR and build a dataframe
    text = ocr_extraction.extract_bulk(read_path, n=3)
    ocr_df = pd.DataFrame(text, index=np.arange(3)) #insert column headers here

    #extract job_id from image_id
    ocr_df["job_id"] = ocr_df["image_id"].apply(lambda x: x.split(".")[0])
    
    #basic cleaning to strip additional characters 
    ocr_df["clean_text"] = ocr_df["ocrd_text"].apply(ocr_evaluation.strip_additional_characters)
    
    #evaluate ocr quality
    ocr_df["plain_accuracy"] = ocr_df["ocrd_text"].apply(ocr_evaluation.calculate_accuracy)
    ocr_df["clean_accuracy"] = ocr_df["clean_text"].apply(ocr_evaluation.calculate_accuracy)
    
    #iteratre through the dataset, identify poor quality ocr, preprocess images & perform ocr again
    ocr_df = ocr_evaluation.update_ocr(ocr_df)

    #save the final dataframe to a csv
    ocr_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()