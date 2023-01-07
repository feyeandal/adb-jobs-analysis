import gensim
import keras
import logging
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
from keras.preprocessing.text import Tokenizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from urllib.parse import unquote

nltk.download('punkt')

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

# Reading TopJobs Data

def read_topjobs_data(data_path):
    '''Composes the data sample by combining the original Topjobs data with manually annotated tags'''

    # Reading the original Topjobs dataset
    data_full = pd.read_excel(data_path)

    return data_full

# Appending the OCR Outputs to sample data

def append_ocr_output(ocr_path, sample):
    '''Appends the OCR output of images to the data sample'''

    # Setting the topjobs id code as the index of the sample
    sample.set_index('job_code', drop=False)

    # Reading the OCR outputs
    sample_ocr_output = pd.read_csv(ocr_path)

    # Renaming the common index for consistency
    sample_ocr_output = sample_ocr_output.rename(columns={'vacancy_id': 'job_code'})

    # Composing a single dataset by appending OCR outputs to the sample
    sample = pd.merge(left=sample, right=sample_ocr_output, how='right', on='job_code').dropna()

    # Lowercasing the OCR output
    sample['tj_desc'] = sample['clean_text'].str.lower()
    print (sample.shape)

    # Dropping unnecessary columns
    sample_ocr_output = sample_ocr_output.drop(columns=['ocrd_text', 'plain_accuracy', 'clean_accuracy'])

    return sample

# Calculating the Lockdown Status for Job Posting

def calculate_lockdown_status(sample, lockdown_date_range):
    '''Returns True if the start date of the job posting falls within the date range of lockdowns provided.
    Returns False otherwise.'''

    lockdown_lowerbound = lockdown_date_range[0] <= sample.start_date
    lockdown_upperbound = sample.start_date <= lockdown_date_range[1]
    sample['lockdown_status'] = (lockdown_lowerbound & lockdown_upperbound)
    sample.groupby(by=['lockdown_status']).size()

    return sample

# Calculating whether the Job Posting Mentions Work From Home Opportunities

def calculate_wfh_mention_status(sample):
    '''Returns True if the job posting mentions the phrase work from home.
    Returns False otherwise'''

    sample['wfh_status'] = np.where(sample['tj_desc'].str.contains('work from home'),True, False)

    return sample

# Renaming Relevant Columns, Adding Status Columns, and Removing Unnecessary Columns

def prepare_sample(sample, lockdown_date_range):
    '''Prepares the sample for the task at hand by renaming relevant columns,
    adding lockdown and Work From Home mention status,
    and removing unneeded columns.'''

    sample = calculate_lockdown_status(sample, lockdown_date_range)
    sample = calculate_wfh_mention_status(sample)

    sample = sample.rename(columns={'job_code': 'tj_code', 'job_title': 'tj_title', })
    sample = sample.drop(columns=['job_description', 'remark', 'functional_area', 'expiry_date', 'image_string', 'image_source', 'image_code', 'image_url'])
    
    return sample

# Creating ONET Corpus

def create_onet_corpus(onet_data, onet_corpus_path):
    '''Creates a combined corpus where each data item consists of the onet occupation titles and alternate titles associated with each job separated by spaces.'''
    
    onet_data['onet_family'] = onet_data['onet_code'].str.slice(stop=2)
    onet_corpus = onet_data.onet_title + ' ' + [' '.join(titles) for titles in onet_data.onet_title_alt]

    onet_corpus.to_csv(onet_corpus_path, index=False)

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
    print(len(sample_title))
    print(sample_title[0])
    print(sample_title[2])
    sample_title_array = np.array(sample_title).reshape(1,-1)
    print(sample_title_array.shape)
    print(sample_title_array[0,3])
    print(sample_title_array[0])
    sample_tfidf_title = tfidf_vect.transform(sample_title_array)

    # Generating tf-idf vectors for job descriptions of Topjpbs data extracted from images using OCR
    sample_desc = sample.tj_desc
    sample_tfidf_desc = tfidf_vect.transform(sample_desc)

    return sample_tfidf_title, sample_tfidf_desc

# Transforming the Sample

# Calculating Cosine Similarity with Different Weights
def calculate_cosine_similarity(onet_tfidf, sample_tfidf_title, sample_tfidf_desc, onet_data, sample):
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

    del wl_title, we_title, wl_desc, we_desc

    return sample_comb

# Matching ONET Categories to Job Postings

def get_onet_matches(sample, sample_comb, onet_data, matches_path):
    '''Combines the Topjobs data sample with details of the ONET occupation matched to each vacancy posting via tf-idf.'''

    # Creating new dataframe
    matches = pd.DataFrame(index=sample.tj_code)

    # Adding the ONET occupation code and family code of the matched occupation to each Topjobs posting
    for job in sample.tj_code:
        code = sample_comb.loc[job, sample_comb.columns].idxmax()
        broad = code[0:6]+'0'
        family = code[0:2]

        matches.loc[job, 'onet_code'] = code
        matches.loc[job, 'onet_broad_occupation_category'] = broad
        matches.loc[job, 'onet_family'] = family

    matches = matches.reset_index()
    matches = matches[matches['onet_family'].notna()]

    # Composing a single dataset by appending useful data from the Topjobs dataset to matches dataset
    matches = pd.merge(left=matches, right=sample, on='tj_code')
    matches = pd.merge(left=matches, right=onet_data[['onet_code', 'onet_title', 'onet_desc']], on='onet_code')

    # Cleaning the title column
    matches.tj_title = [unquote(str(title)) for title in matches.tj_title]
    matches.tj_title = [re.sub('\+', ' ', title) for title in matches.tj_title]

    matches = matches.set_index('tj_code', drop=False)

    matches.to_csv(matches_path, index=False)

    return matches

################################################

# Main Function
def main(data_path, occ_path, alt_path, tech_path, ocr_output_path, lockdown_date_range, onet_corpus_path, matches_path):
    
    logging.info('Reading onet data')
    onet_data = read_onet_data(occ_path, alt_path, tech_path)

    logging.info('Reading topjobs data')
    sample = read_topjobs_data(data_path)

    logging.info('Appending ocr output')
    sample = append_ocr_output(ocr_output_path, sample)

    logging.info('Preparing sample')
    sample = prepare_sample(sample, lockdown_date_range)

    logging.info('Onet corpus')
    onet_corpus = create_onet_corpus(onet_data, onet_corpus_path)

    logging.info('TFIDF')
    tfidf_vect, onet_tfidf = create_tf_idf_vector(onet_corpus)

    logging.info('vectorizing')
    sample_tfidf_title, sample_tfidf_desc = vectorize_sample(sample, tfidf_vect)
    
    logging.info('cosine similarity')
    sample_comb = calculate_cosine_similarity(onet_tfidf, sample_tfidf_title, sample_tfidf_desc, onet_data, sample)

    logging.info('getting matches')
    matches = get_onet_matches(sample, sample_comb, onet_data, matches_path)
    
    return (matches)
