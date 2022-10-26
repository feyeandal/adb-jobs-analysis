# Introduction
The objective of this code is to conduct a job and skills analysis on TopJops data. The code is broken into modules as follows.
- OCR Extraction of text from TopJops vacancy advertisement images
- Classification of vacancy advertisement text into ONET occupation categories
- Skills analysis of vacancy advertisement text using topic modelling

# Inputs
The input files are available to download at the following locations.
- [*TopJobs Images*](https://lirneasia2-my.sharepoint.com/:f:/g/personal/merl_lirneasia_net/EkPiEnmE0oJOhQFiWSIgkxoBhtzlJN3wBAGFYQa8V4qIxg?e=paDT1F)
- [*TopJobs Metadata*](https://docs.google.com/spreadsheets/d/1RzKhN8WsJd46Exp0xpUZTFA7Vx1CkSbV/edit?usp=sharing&ouid=104722114468116715910&rtpof=true&sd=true)
- [*ONET Occupation Titles*](https://drive.google.com/file/d/10nz9nUwI40hnnjA98V0VlPZweGcB2FNS/view?usp=sharing)
- [*ONET Alternate Occupation Titles*](https://drive.google.com/file/d/162KI9FzY0WtHITY6oyvnEScql3tKy19m/view?usp=sharing)
- [*Technologies Associated with ONET Occupations*](https://drive.google.com/file/d/1H2FFbkPfAe27WtDoALLmBuFmBQgtc-2g/view?usp=sharing)
- [*TopJobs Manually Annotated Tags*](https://drive.google.com/file/d/1aHbwE212BWWWE0i1Aem8deDLEKQACWIs/view?usp=sharing)

# Setting Up
Your device can be set up to run this code by following the given steps.
- Download and install Tesseract on your device.
- Download the [Requirements File](https://github.com/LIRNEasia/adb-jobs-analysis/blob/b53aa3bc076b4df0072c07d0d36b547eac1a89ff/requirements.txt) and install the dependencies.

# Running the Code
The Python scripts that should be run to carry out each task are as follows.
    - *OCR Extraction Process:* ocr_pipeline.py
    - *ONET Classification Process:* onet_classification.py
    - *Skills Analysis (LDA):* skills_analysis.py
    - *Skills Analysis (Top2Vec): topic_modeling_top2vec.py
    - *Combined Pipeline:* overall_pipeline.py

## OCR Extraction
The OCR extraction process is handled by the following set of scripts.
- ocr_extraction.py
- ocr_evaluation.py
- ocr_pipeline.py

In order to run the OCR extraction, the following filepaths in *config.yaml* should be updated to reflect the relevant file locations on your device.
- *image_path:* Path to the folder of TopJobs images on your device, unzipped and added to a single folder
- *ocr_output_path:* The path to which you prefer your output file of OCR text to be saved
- *ocr_model_path:* The path to the OCR image preprocessing model on your device

## ONET Classification
The script *onet_classification.py* handles the ONET classification process. 

To run the ONET classification, the following filepaths in *config.yaml* should be updated to reflect the relevant file locations on your device.
- *data_path:* Path to the TopJobs metadata file on your device
- *occ_path:* Path to the ONET occupation titles file on your device
- *alt_path:* Path to the ONET alternate occupation titles file on your device
- *tech_path:* Path to the file of ONET technologies associated with occupation titles on your device
- *tags_path:* Path to the manually annotated TopJobs dataset on your device
- *ocr_output_path:* Path to which the OCR output file generated in *ocr_pipeline.py* has been saved
- *onet_corpus_path:* Path to which you prefer your output file of ONET data to be saved
- *matches_path:* Path to which you prefer your output file of ONET occupations matched to TopJobs vacancies to be saved

## Skills Analysis (LDA)
LDA-based skills analysis is handled by the script *skills_analysis.py*.

To run the LDA skills analysis, the following filepath in *config.yaml* should be updated to reflect the relevant file location on your device.
- *ocr_output_path:* Path to which the OCR output file generated in *ocr_pipeline.py* has been saved

## Skills Analysis (Top2Vec)
Top2Vec-based skills analysis is handled by the script *topic_modeling_top2vec.py*.

To run the Top2Vec skills analysis, the following filepath in *config.yaml* should be updated to reflect the relevant file location on your device.
- *ocr_output_path:* Path to which the OCR output file generated in *ocr_pipeline.py* has been saved
- *wordclouds_path:* Path to which you prefer your output wordcloud file to be saved

## Combined Pipeline
The script *overall_pipeline.py* handles the combined pipeline consisting of OCR extraction, ONET classification, and skills analysis.

To run the overall pipeline, the following filepaths in *config.yaml* should be updated to reflect the relevant file locations on your device.
- *image_path:* Path to the folder of TopJobs images on your device, unzipped and added to a single folder
- *data_path:* Path to the TopJobs metadata file on your device
- *occ_path:* Path to the ONET occupation titles file on your device
- *alt_path:* Path to the ONET alternate occupation titles file on your device
- *tech_path:* Path to the file of ONET technologies associated with occupation titles on your device
- *tags_path:* Path to the manually annotated TopJobs dataset on your device
- *ocr_output_path:* The path to which you prefer your output file of OCR text to be saved
- *onet_corpus_path:* Path to which you prefer your output file of ONET data to be saved
- *matches_path:* Path to which you prefer your output file of ONET occupations matched to TopJobs vacancies to be saved
- *wordclouds_path:* Path to which you prefer your output wordcloud file to be saved
- *ocr_model_path:* The path to the OCR image preprocessing model on your device
