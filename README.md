# Introduction
Online job portals (OJPs) have quickly become an important source of labor market information. They are able to provide granular, near-real-time information about the demand for jobs and skills in labor markets. This is in contrast with more conventional data sources, such as labor force surveys, which, while having great value in terms of ensuring coverage, rigour and comparability, fail to capture the complexity, the variability, and the pace of change in labour markets. 

However, despite these advantages, several challenges need to be overcome before OJPs can be reliably used to draw conclusions on labour markets. First, OJP data is usually unrepresentative of the full job market. Not all vacancies are advertised, and even among the advertised, there are differences in coverage from one OJP to another based on their target market segment, language, the approach used to collect ads, etc.  The second is data quality. Given that the data generated on OJPs aren’t collected with research objectives in mind, there are no common standards for vacancy formats, schemas, and job classifications. Even within the same country, there can be significant differences in the nature, amount, and quality of data captured by different job portals. Thirdly, the legal and ethical frameworks for obtaining and utilizing job portal data are not always clearly established. 

Therefore, it is important to demonstrate potential solutions to these problems via pilot studies before OJPs are used as a data source for real-life decision-making.  As such, we have partnered with [http://topjobs.lk/](http://topjobs.lk/) to demonstrate a use case within the Sri Lankan context. Specifically, we have chosen to investigate the impact of COVID-19 on the Sri Lankan job market in two major categories of jobs available on TopJobs: IT/Software and Hospitality. For these two categories of jobs, we aim to answer two broad questions.

1. What was the impact of COVID-19 on the number/variety of jobs posted?
2. What are the key differences in skills and other requirements sought by employers before and after the COVID-19 outbreak?

To that end, this repository is organized into modules which achieve three key tasks.

1. ***Developing a pipeline to process images into machine-readable text***. Topjobs displays its vacancies in the form of images which are not readily analyzable using computational techniques. With the help of image processing, optical character recognition (OCR) and natural language processing (NLP), we set up a data processing pipeline which takes job vacancies as inputs and produces machine-readable text that can be used for labour market analysis.
2. ***Decomposing high-level job categories into meaningful job families***. Since TopJobs data only includes broad categories of jobs, we experiment with alternative ways of further categorizing occupations. Here we map TopJobs’ vacancies into O*NET’s classification of jobs (O*NET is a system which groups similar occupations based on work performed and on required skills, education, training, and credentials) so that each vacancy is assigned into a family of jobs
3. ***Using machine learning to extract skills***. Given the unstructured nature of vacancy data found on TopJobs, skills are not readily extractable. Therefore, we are using unsupervised machine learning techniques, Topic Modeling to be specific, to extract skills from a given vacancy, which enables us to compare the changes in skills demanded by employers before and after the onset of the COVID-19 pandemic

# Technical Requirements
- ***Disk space:*** 8GB
- ***Python Version:*** Python 3[^1]

[^1]: This project was tested on Python versions 3.8.10. and 3.10.0. While the code may work for other Python versions, we recommend using a version it is already tested on.

# Data
The input data usde for the study are available to download at the following locations.

## TopJobs Images
Job vacancies in TopJobs are posted in the form of jpg or png images. The full set of image vacancies (for the 2018-2021 period) and corresponding meta data have been shared separately. A small subset of image vacancies from the IT/Software sector of TopJobs is linked here (as a zipped folder) for purposes of illustration/replication. Any subset of images from the larger database can be executed using this method.
The zipped images are available [*here*](https://lirneasia2-my.sharepoint.com/:u:/g/personal/merl_lirneasia_net/Ef4n8KNHTIlBhHwKshd22J0BOOJdwgu2sktWQaFCGL8WBQ?e=SPGHec). All your data should be stored in a single folder to replicate the study.

## TopJobs Metadata
This is a datasheet consisting of metadata related to each of the job vacancy postings mentioned above, obtained from TopJobs. Details such as the time of posting, job title, and the functional area of the job are included in this file.
The metadata file is available [*here*](https://lirneasia2-my.sharepoint.com/:x:/g/personal/merl_lirneasia_net/EW4xiospTGBYyZITmNzjXy0BIJWwe0r2b32z1zWZ3U_eYw?e=VpEcrG).

## ONET Occupation Titles
Occupation title for each of the O*NET occupation categories are included in this dataset, which is available to download [*here*](https://raw.githubusercontent.com/LIRNEasia/adb-jobs-analysis/main/data/onet_data/onet_occ_titles.txt).

## ONET Alternate Occupation Titles
This file includes a list of alternative occupation titles that are commonly used for each of the O*NET occupations. The dataset is available [*here*](https://raw.githubusercontent.com/LIRNEasia/adb-jobs-analysis/main/data/onet_data/onet_alt_titles.txt).

## Technologies Associated with ONET Occupations
Technologies that are commonly utilized in each O*NET occupation category, and therefore are associated with the occupation category, are included in this dataset, which is available [*here*](https://raw.githubusercontent.com/LIRNEasia/adb-jobs-analysis/main/data/onet_data/onet_tech_skills.txt).

## TopJobs Manually Annotated Tags
This is a sample of 500 job vacancy postings that was manually annotated with corresponding O*NET categories. The dataset is available for download  [*here*](https://drive.google.com/file/d/1aHbwE212BWWWE0i1Aem8deDLEKQACWIs/view?usp=sharing).

## Image Upscaling Model
This model is used to improve the performance of optical character recognition by enhancing the resolution of images. The trained model is available [*here*](https://lirneasia2-my.sharepoint.com/:u:/g/personal/vihanga_lirneasia_net/ERXGlwalEQBPjXibnPNyXeABPiU5VkIo6VMe116UDbMtLA?e=xKHnCJ).

# Setting Up the Dev Environment
Your device can be set up to run this analysis by following the given steps. We recommend using a *NIX based OS for running this analysis (e.g., Linux distribution or MacOS)

1. Clone the GitHub Repository (the example below assumes that github authentication is performed using ssh)
```
$ git clone git@github.com:LIRNEasia/adb-jobs-analysis.git
```

2. This project was developed and tested on Python 3.8.10. It is advised to set up a Python virtual environment with the same Python version. Virtual environments can be created using either `venv`, `conda`, or `pyenv`. We recommend using `pyenv` as it enables managing different python versions as well (more info on pyenv can be found [here](https://github.com/pyenv/pyenv)


3.  We use the Google's Tesseract engine and its Python wrapper pytesseract for optical character recognition. In order to use that you have to download and install Tesseract on your device. You may follow one of the following tutorials to install Tesseract, based on your OS.
    - [Windows Users](https://codetoprosper.com/tesseract-ocr-for-windows)
    - [Mac Users](https://www.oreilly.com/library/view/building-computer-vision/9781838644673/95de5b35-436b-4668-8ca2-44970a6e2924.xhtml)
    - [Ubuntu Users](https://linuxhint.com/install-tesseract-ocr-linux/)
    
4. Once the virtual environment is created with Python 3.8.10, please activate it. Then, we can install the Python packages necessary to run this analysis using the provided `requirements.txt` file using the following command. 

```
$ pip install -r requirements.txt
```

5. Download the Spacy model *en_core_web_sm* required for LDA-based topic analysis. This can be done by running the following code snippet from your command line.
```
$ python -m spacy download en_core_web_sm
```

Once you successfully complete these steps, you have your developement environment setup to replicate this analysis. 

# Scripts
The Python script *main.py* handles the overall process. To run a selected module or a set of modules, the variable *process_name* in the configuration file *config.yaml* should be updated as follows.
    - *OCR Extraction Process:* ocr_extraction
    - *ONET Classification Process:* onet_classification
    - *ONET Evaluation Process:* onet_evaluation
    - *Skills Analysis (LDA):* skills_analysis_lda
    - *Skills Analysis (Top2Vec):* skills_analysis_top2vec

## OCR Extraction
The purpose of OCR extraction is to identify and extract the textual content in all images in a given folder.

The OCR extraction process is handled by the following set of scripts.
- ocr_extraction.py
- ocr_evaluation.py
- ocr_pipeline.py

In order to run the OCR extraction, the following filepaths in *config.yaml* should be updated to reflect the relevant file locations on your device.
- *image_path:* Path to the folder of TopJobs images on your device, unzipped and merged into a single folder
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

# Replicating the Study
In order to replicate this study, the following steps could be followed.
- Download all the data mentioned in the **Data** section to your device.
- Unzip the TopJobs image data and merge all the subfolders into a single folder.
- Set up your device following the steps mentioned in the **Setting Up** Section.
- Unzip the set of TopJobs Image folders and merge all the images included into a single folder.
- Update all filepaths and folderpaths in the file **config.yaml** to the relevant paths on your device.
- Set up a Python virtual environment. You may follow the instructions given [*here*](https://dssg.github.io/hitchhikers-guide/curriculum/setup/software-setup/setup_windows/) for that.
- Install all packages mentioned in **requirements.txt**.
- Set the variable *script_name* in the file **config.yaml** to reflect the process you wish to run, as mentioned in the **Scripts** section.
    - *OCR Extraction Process:* ocr_extraction
    - *ONET Classification Process:* onet_classification
    - *ONET Evaluation Process:* onet_evaluation
    - *Skills Analysis (LDA):* skills_analysis_lda
    - *Skills Analysis (Top2Vec):* skills_analysis_top2vec
- Run the file **main.py** from your virtual environment, replacing *<path/to/main.py>* with the filepath to main.py on your device, and *<path/to/config/file>* with the filepath to the yaml config file on your device.
    - ```python path/to/main.py -c path/to/config/file```
- You may access the output files by visiting the filepaths you have specified for them in the file **config.yaml**, through your file explorer.
