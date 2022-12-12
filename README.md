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

The script `main.py` runs all the modules in the project. The script executes the following:

- OCR to extract text from the advert images
- ONET classification to map the adverts to the ONET job classification
- Skills analysis to extract specific skills required by the job adverts.

To parameters needed by the script is set through a configuration file. The `example_config.yaml` includes details on how to setup the configuration file. Once the configuration is setup, the code can be executed from the root folder of the project using the following command:

```
$ python src/main.py -c path/to/the/config/file
```

<!-- # Running the Code
The Python script *main.py* handles the overall process. To run a selected module or a set of modules, the variable *process_name* in the configuration file *config.yaml* should be updated as follows.
    - *OCR Extraction Process:* ocr_extraction
    - *ONET Classification Process:* onet_classification
    - *ONET Evaluation Process:* onet_evaluation
    - *Skills Analysis (LDA):* skills_analysis_lda
    - *Skills Analysis (Top2Vec):* skills_analysis_top2vec
    - *Combined Pipeline:* overall_pipeline -->

## OCR Extraction
The purpose of OCR extraction is to identify and extract the textual content in all images in a given folder.

The OCR extraction process is handled by the following set of scripts.
- ocr_extraction.py
- ocr_evaluation.py
- ocr_pipeline.py

In order to run the OCR extraction, the following filepaths in *config.yaml* should be updated to reflect the relevant file locations on your device.
- *image_path:* Path to the folder of TopJobs images on your device, unzipped and added to a single folder
- *ocr_output_path:* The path to which you prefer your output file of OCR text to be saved
- *ocr_model_path:* The path to the OCR image preprocessing model on your device

## ONET Classification
The ONET classification aims to identify and annotate all job vacancy postings in a given topjobs dataset with the relevant ONET Occupation codes.

The script *onet_classification.py* handles the ONET classification process. 

To run the ONET classification, the following filepaths in *config.yaml* should be updated to reflect the relevant file locations on your device.
- *data_path:* Path to the TopJobs metadata file on your device
- *occ_path:* Path to the ONET occupation titles file on your device
- *alt_path:* Path to the ONET alternate occupation titles file on your device
- *tech_path:* Path to the file of ONET technologies associated with occupation titles on your device
- *ocr_output_path:* Path to which the OCR output file generated in *ocr_pipeline.py* has been saved
- *onet_corpus_path:* Path to which you prefer your output file of ONET data to be saved
- *matches_path:* Path to which you prefer your output file of ONET occupations matched to TopJobs vacancies to be saved

## ONET Evaluation
The objective of the ONET evaluation is to evaluate the performance of the ONET Classification process using a TopJobs data sample of 500 vacancy postings which is manually annotated with ONET categories.

This process is handled by the script *onet_evaluation.py*.

To run the ONET evaluation, the following filepaths in *config.yaml* should be updated to reflect the relevant file location on your device.

- *tags_path:* Path to the manually annotated TopJobs dataset on your device
- *matches_path:* Path to which you the matches output file generated in *onet_classification.py* has been saved

## LDA-Based Skills Analysis
The objective of this process is to identify the dominant sets of skills mentioned in a given vacancy dataset using LDA topic modelling.

LDA-based skills analysis is handled by the script *skills_analysis.py*.

To run the LDA skills analysis, the following filepath in *config.yaml* should be updated to reflect the relevant file location on your device.
- *ocr_output_path:* Path to which the OCR output file generated in *ocr_pipeline.py* has been saved

## Top2Vec-Based Skills Analysis
This process aims to identify the dominant sets of skills in a given vacancy dataset using Top2Vec topic modelling.

Top2Vec-based skills analysis is handled by the script *topic_modeling_top2vec.py*.

To run the Top2Vec skills analysis, the following filepath in *config.yaml* should be updated to reflect the relevant file location on your device.
- *ocr_output_path:* Path to which the OCR output file generated in *ocr_pipeline.py* has been saved
- *wordclouds_path:* Path to which you prefer your output wordcloud file to be saved

## Combined Pipeline
The combined pipeline runs the whole process of OCR extraction, ONET classification, LDA-based skills analysis, and Top2Vec-based skills analysis.

To run the overall pipeline, the following filepaths in *config.yaml* should be updated to reflect the relevant file locations on your device.
- *image_path:* Path to the folder of TopJobs images on your device, unzipped and added to a single folder
- *data_path:* Path to the TopJobs metadata file on your device
- *occ_path:* Path to the ONET occupation titles file on your device
- *alt_path:* Path to the ONET alternate occupation titles file on your device
- *tech_path:* Path to the file of ONET technologies associated with occupation titles on your device
- *ocr_output_path:* The path to which you prefer your output file of OCR text to be saved
- *onet_corpus_path:* Path to which you prefer your output file of ONET data to be saved
- *matches_path:* Path to which you prefer your output file of ONET occupations matched to TopJobs vacancies to be saved
- *wordclouds_path:* Path to which you prefer your output wordcloud file to be saved
- *ocr_model_path:* The path to the OCR image preprocessing model on your device
