import gensim
import keras
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re
import seaborn as sn
import tensorflow as tf

from gensim.models import word2vec
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
from keras import backend as K
from keras.constraints import maxnorm
from keras.models import Sequential,Model,load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from urllib.parse import unquote

# Reading ONET Data

def read_onet_data(occ_path, alt_path, tech_path):
    '''Reads ONET occupation titles, alternate occupation titles, and technolgies associated with occupations from relevant sources.
    Returns a combined dataset consisting of all of these details.'''

    # Reading the list of ONET occupation titles
    onet_occ = pd.read_csv(occ_path, sep='\t')

    # Reading the list of ONET alternate occupation titles
    onet_alt = pd.read_csv(alt_path, sep='\t').groupby(by='onet_code')['onet_title_alt'].apply(list)
    
    # Reading the list of technologies associated with ONET occupations
    onet_tech = pd.read_csv(tech_path, sep='\t').groupby(by='onet_code')['onet_tech'].apply(list)

    # Compposing a single dataset with ONET codes, occupation titles, alternate titles, and technologies associated
    onet_data = pd.merge(left=onet_occ, right=onet_alt, how='left', on='onet_code')

    onet_data = pd.merge(
        left=onet_data, right=onet_tech, how='left', on='onet_code') \
        .fillna('')

    del onet_occ, onet_alt, onet_tech

    return onet_data

# Reading the Evaluation Corpus

def compose_data_sample(data_path, tags_path):
    '''Composes the data sample by combining the original Topjobs data with manually annotated tags'''

    # Reading the original Topjobs dataset
    data_full = pd.read_excel(data_path)

    # Reading the manually annotated tags for the data sample
    sample_tags = pd.read_csv(tags_path)

    # sample_tags['image_drive_url'] = image_path + sample_tags['file_name']
    sample_tags['job_code'] = sample_tags.file_name.replace(to_replace='[^0-9]', value='', regex=True).astype(int)
    
    # Composing a single dataset by appending manually annotated tags to the Topjobs dataset
    sample = pd.merge(left=sample_tags, right=data_full, how='left', on='job_code')
    sample['tags'] = sample[['tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8', 'tag_9', 'tag_10']].values.tolist()
    sample = sample.drop(columns=['tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8', 'tag_9', 'tag_10'])

    del sample_tags, data_full

    return sample

# Appending the OCR Outputs to sample data

def append_ocr_output(ocr_path, sample):
    '''Appends the OCR output of images to the data sample'''

    # Setting the topjobs id code as the index of the sample
    sample.set_index('job_code', drop=False)

    # Reading the OCR outputs
    sample_ocr_output = pd.read_csv(ocr_path)

    # Renaming the common index for consistency
    sample_ocr_output = sample_ocr_output.rename(columns={'job_id': 'job_code'})

    # Dropping unnecessary columns
    sample_ocr_output = sample_ocr_output.drop(columns=['ocrd_text', 'plain_accuracy', 'clean_accuracy'])

    # Composing a single dataset by appending OCR outputs to the sample
    sample = pd.merge(left=sample, right=sample_ocr_output, how='left', on='job_code')

    # Lowercasing the OCR output
    sample['tj_desc'] = sample['ocr_output'].str.lower()

    return sample

# Calculating the Lockdown Status for Job Posting

def calculate_lockdown_status(sample, lockdown_date_range):
    '''Returns 1 if the start date of the job posting falls within the date range of lockdowns provided.
    Returns 0 otherwise.'''

    lockdown_lowerbound = lockdown_date_range[0] <= sample.start_date
    lockdown_upperbound = sample.start_date <= lockdown_date_range[1]
    sample['lockdown_status'] = (lockdown_lowerbound & lockdown_upperbound)
    sample.groupby(by=['lockdown_status']).size()

    return sample

# Calculating whether the Job Posting Mentions Work From Home Opportunities

def calculate_wfh_mention_status(sample):
    '''Returns 1 if the job posting mentions the phrase work from home.
    Returns 0 otherwise'''

    sample['wfh_status'] = np.where(sample['tj_desc'].str.contains('work from home'),1, 0)

    return sample

# Renaming Relevant Columns, Adding Status Columns, and Removing Unnecessary Columns

def prepare_sample(sample, lockdown_date_range):
    '''Prepares the sample for the task at hand by renaming relevant columns,
    adding lockdown and Work From Home mention status,
    and removing unneeded columns.'''

    sample = calculate_lockdown_status(sample, lockdown_date_range)
    sample = calculate_wfh_mention_status(sample)

    sample = sample.rename(columns={'job_code': 'tj_code', 'job_title': 'tj_title', })
    sample = sample.drop(columns=['image_drive_url', 'job_description', 'remark', 'functional_area', 'expiry_date', 'image_string', 'image_source', 'image_code', 'image_url', 'start_date'])
    
    return sample

# Creating ONET Corpus

def create_onet_corpus(onet_data):
    '''Creates a combined corpus where each data item consists of the onet occupation titles and alternate titles associated with each job separated by spaces.'''
    onet_data['onet_family'] = onet_data['onet_code'].str.slice(stop=2)
    onet_corpus = onet_data.onet_title + ' ' + [' '.join(titles) for titles in onet_data.onet_title_alt]

    onet_corpus.to_csv(folder_path+'data/outputs/onet_corpus.csv', index=False)

    return onet_corpus

# Tokenizing and stemming English text

def nltk_tokenizer_stemmer(text):
    '''Tokenizes and stems the words in a given body of text.'''

    # Dividing text into tokens
    tokens = [word for word in word_tokenize(text)]

    # Stemming word tokens
    stems = [PorterStemmer().stem(word) for word in tokens]

    return stems

# Fitting the tf-idf Vectorizer on the Reference Corpus
def create_tf_idf_vector(onet_corpus):
    '''Creates and trains a tf-idf vectorizer on the ONET corpus consisting of occupation titles and alternate titles.'''
    
    # Creating the vectorizer
    tfidf_vect = TfidfVectorizer(tokenizer=nltk_tokenizer_stemmer, stop_words='english')

    # Training the tf-idf vectorizer on the ONET corpus
    onet_tfidf = tfidf_vect.fit_transform(onet_corpus)

    return tfidf_vect, onet_tfidf

# Vectorizing Titles and Descriptions Separately

def vectorize_sample(sample, tfidf_vect):
    '''Generates tf-idf vectors for the data sample.'''

    # Generating tf-idf vectors for job titles of Topjpbs data
    sample_title = [unquote(str(title)) for title in sample.tj_title]
    sample_title = [re.sub('\+', ' ', title) for title in sample_title]
    sample_tfidf_title = tfidf_vect.transform(sample_title)

    # Generating tf-idf vectors for job descriptions of Topjpbs data extracted from images using OCR
    sample_desc = sample.tj_desc
    sample_tfidf_desc = tfidf_vect.transform(sample_desc)

    return sample_tfidf_title, sample_tfidf_desc

# Transforming the Sample

# Calculating Cosine Similarity with Different Weights
def calculate_cosine_similarity(onet_tfidf, sample_tfidf_title, sample_tfidf_desc):
    '''Calculates the cosine similarity between each of the ONET occupations and each of the Topjobs vacancy posting.
    Considers occupation title and alternative titles for ONET occupations.
    Considers job title and job description for Topjobs data.'''

    wl_title = 0.6
    we_title = 1
    wl_desc = 1-wl_title
    we_desc = 1

    # Calculating similarity between each entry in the ONET corpus and Topjobs job titles
    sample_title = linear_kernel(sample_tfidf_title, onet_tfidf)

    # Calculating similarity between each entry in the ONET corpus and Topjobs job descriptions
    sample_desc = linear_kernel(sample_tfidf_desc, onet_tfidf)

    # Calculating the combined similarity value for each Topjobs posting
    sample_comb = pd.DataFrame(
        data=(sample_title**we_title)*wl_title + (sample_desc**we_desc)*wl_desc,
        columns=onet_data.onet_code,#onet_data.onet_family.unique(),
        index=sample.tj_code)

    # sample_comb.to_csv(folder_path+'data/outputs/sample_comb.csv', index=False)

    del wl_title, we_title, wl_desc, we_desc

    return sample_comb

# Matching ONET Categories to Job Postings

def get_onet_matches(sample):
    '''Combines the Topjobs data sample with details of the ONET occupation matched to each vacancy posting via tf-idf.'''

    # Creating new dataframe
    matches = pd.DataFrame(index=sample.tj_code)

    # Adding the ONET occupation code and family code of the matched occupation to each Topjobs posting
    for job in sample.tj_code:
        code = sample_comb.loc[job, sample_comb.columns].idxmax() #.str.startswith(family)].idxmax()
        family = code[0:2]
        matches.loc[job, 'onet_code'] = code
        matches.loc[job, 'onet_family'] = family

    matches = matches.reset_index()
    matches = matches[matches['onet_family'].notna()]

    # Composing a single dataset by appending useful data from the Topjobs dataset to matches dataset
    matches = pd.merge(left=matches, right=sample[['tj_code', 'tj_title', 'tj_desc', 'tags', 'lockdown_status', 'wfh_status']], on='tj_code')
    matches = pd.merge(left=matches, right=onet_data[['onet_code', 'onet_title', 'onet_desc']], on='onet_code')

    # Cleaning the title column
    matches.tj_title = [unquote(str(title)) for title in matches.tj_title]
    matches.tj_title = [re.sub('\+', ' ', title) for title in matches.tj_title]

    # matches_title = matches.groupby('onet_family').size()
    matches = matches.set_index('tj_code', drop=False)

    return matches

# Evaluating Matches

def evaluate_matches(matches):
    '''Evaluates the performance of the tf-idf vectorizer by generating a confusion matrix between manually anotated ONET categories and those annotated via tf-idf.'''

    for job in matches.tj_code:
        tags = matches.at[job,'tags']
        tag_families = [int(str(tag)[0:2]) for tag in tags if pd.notnull(tag)]
        tag_families = (tag_families + [None]*10)[:10]
        onet_family = matches.at[job,'onet_family']

        matches.loc[job, 'tag_families'] = [[tag_families]]
        matches.loc[job, 'first_tag_family'] = str(matches.loc[job, 'tag_families'][0][0])
        matches.loc[job, 'match_value'] = [int(onet_family) in tag_families]

    # matches.groupby(['wfh_status', 'lockdown_status']).size()

    matches.to_csv(folder_path+'data/outputs/sample_matches.csv', index=False)

    # Generating the confusion matrix
    confusion_matrix = pd.crosstab(matches.first_tag_family, matches.onet_family, rownames=['Actual'], colnames=['Predicted'])
    print (confusion_matrix)

    # Generating the heatmap for the confusion matrix
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()

    return confusion_matrix

################################################

# Path to the folder where project data is saved
folder_path = '/content/drive/MyDrive/LIRNEasia/ADB Project/'

# Path to the full Topjobs dataset
data_path = folder_path+'data/data_full.xlsx'

# Path to the dataset of ONET occupation titles
occ_path = folder_path + 'data/onet_occ_titles.txt'

# Path to the dataset of ONET alternate occupation titles
alt_path = folder_path+'data/onet_alt_titles.txt'

# Path to the dataset of technologies associated with ONET occupations
tech_path = folder_path+'data/onet_tech_skills.txt'

# Path to the dataset of manually annotated tags for the Topjobs data sample
tags_path = folder_path+'data/cs_sample_tags.csv'

# image_path = folder_path+'data/sample/'

# Path to the OCR outputs for the Topjobs data sample
ocr_path = folder_path+'data/outputs/cs_sample_ocr_output.csv'

# Date range during which lockdown was implemented (ASSUMPTION: All dates beyond 2020-03-01 are considered to be under lockdown)
lockdown_date_range = ['2020-03-01', '2022-12-31']

# Main Function
def main():
    onet_data = read_onet_data(occ_path, alt_path, tech_path)

    sample = compose_data_sample(data_path, tags_path)
    sample = append_ocr_output(ocr_path, sample)
    sample = prepare_sample(sample, lockdown_date_range)

    onet_corpus = create_onet_corpus(onet_data)
    tfidf_vect, onet_tfidf = create_tf_idf_vector(onet_corpus)

    sample_tfidf_title, sample_tfidf_desc = vectorize_sample(sample, tfidf_vect)
    sample_comb = calculate_cosine_similarity(onet_tfidf, sample_tfidf_title, sample_tfidf_desc)

    matches = get_onet_matches(sample)
    confusion_matrix = evaluate_matches(matches)
    return (confusion_matrix)

if __name__ == '__main__':
    main()