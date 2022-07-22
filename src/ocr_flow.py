import numpy as np
import pandas as pd
import ocr_extraction
import ocr_evaluation

read_path = "E:/ADB_Project/code/data/cs_sample"
save_path = "E:/ADB_Project/code/data/pipeline_sample.csv" #change the declaration location 

def main():
    """ reads image from a directory and outputs cleaned text after OCR"""
    #uses the ocr_extraction sub-module to conduct initial OCR
    text = ocr_extraction.extract_bulk(read_path, n=10)
    ocr_df = pd.DataFrame(text, index=np.arange(10)) #insert column headers here
    ocr_df["job_id"] = ocr_df["job_id"].split(".")[0]

    ocr_df = ocr_evaluation.update_ocr(ocr_df)

    ocr_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    main()