import gensim
import keras
import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import re
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

folder_path = '/content/drive/MyDrive/LIRNEasia/ADB Project/'

# Reading Data

onet_occ = pd.read_csv(folder_path+'data/onet_occ_titles.txt', sep='\t')

onet_alt = pd.read_csv(folder_path+'data/onet_alt_titles.txt', sep='\t') \
    .groupby(by='onet_code') \
    .agg({'onet_title_alt': lambda x: x.astype(object)}) \
    .reset_index()

onet_tech = pd.read_csv(folder_path+'data/onet_tech_skills.txt', sep='\t') \
    .groupby(by='onet_code') \
    .agg({'onet_tech': lambda x: x.astype(object)}) \
    .reset_index()

onet_data = pd.merge(left=onet_occ, right=onet_alt, how='left', on='onet_code')

onet_data = pd.merge(
    left=onet_data, right=onet_tech, how='left', on='onet_code') \
    .fillna('')

onet_data.to_csv(folder_path+'data/outputs/onet_data.csv', index=False)

del onet_occ, onet_alt, onet_tech

# Creating the Evaluation Corpus

data_full = pd.read_excel(folder_path+'data/data_full.xlsx')

image_path = folder_path+'data/sample/'
sample_tags = pd.read_csv(folder_path+'data/sample_tags.csv')

sample_tags['image_drive_url'] = image_path + sample_tags['file_name']
sample_tags['job_code'] = sample_tags.file_name.replace(
    to_replace='[^0-9]', value='', regex=True).astype(int)

sample = pd.merge(left=sample_tags, right=data_full, how='left', on='job_code')
sample['tags'] = sample[['tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8', 'tag_9', 'tag_10']].values.tolist()
sample = sample.drop(columns=['tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8', 'tag_9', 'tag_10'])
sample.info()

del sample_tags, data_full

# Getting the OCR Outputs

sample.set_index('job_code', drop=False)

sample_ocr_output = pd.read_csv(folder_path+'data/outputs/sample_ocr_output.csv')
sample_ocr_output = sample_ocr_output.drop(columns=['file_name', 'image_drive_url'])
sample = pd.merge(left=sample, right=sample_ocr_output, how='left', on='job_code')

def clean_text(text):
  #replaces dashes with my chosen delimiter
  nodash = re.sub('.(-+)', ',', text)
  #strikes multiple periods
  nodots = re.sub('.(\.\.+)', '', nodash)
  #strikes linebreaks
  nobreaks = re.sub('\n', ' ', nodots)
  #strikes extra spaces
  nospaces = re.sub('(  +)', ',', nobreaks)
  #strikes *
  nostar = re.sub('.[*]', '', nospaces)
  #strikes new line and comma at the beginning of the line
  flushleft = re.sub('^\W', '', nostar)
  #getting rid of double commas (i.e. - Evarts)
  comma = re.sub(',{2,3}', ',', flushleft)
  #cleaning up some words that are stuck together (i.e. -  Dawes, Manderson)
  return (comma)

sample['tj_desc'] = [clean_text(text) for text in sample.ocr_output]
sample['tj_desc'] = sample['tj_desc'].str.lower()

# Calculating Lockdown Status for Job Posting

sample['lockdown_status'] = sample.start_date >= '2020-01-01'
sample.groupby(by=['lockdown_status']).size()

# Calculating Work From Home Status for Job Posting
sample['wfh_status'] = np.where(sample['tj_desc'].str.contains('work from home'),1, 0)

# Renaming Relevant Columns and Removing Unnecessary Columns

sample = sample.rename(columns={'job_code': 'tj_code', 'job_title': 'tj_title', })
sample = sample.drop(columns=['image_drive_url', 'job_description', 'remark', 'functional_area', 'expiry_date', 'image_string', 'image_source', 'image_code', 'image_url', 'start_date'])

# Creating ONET Corpus

onet_data['onet_family'] = onet_data['onet_code'].str.slice(stop=2)

onet_corpus = onet_data.onet_title + ' ' + \
    [' '.join(titles) for titles in onet_data.onet_title_alt]

onet_corpus.to_csv(folder_path+'data/outputs/onet_corpus.csv', index=False)

# Fitting the tf-idf Vectorizer on the Reference Corpus

def nltk_tokenizer(text):
    tokens = [word for word in word_tokenize(text)]
    stems = [PorterStemmer().stem(word) for word in tokens]
    return stems

tfidf_vect = TfidfVectorizer(tokenizer=nltk_tokenizer, stop_words='english')
onet_tfidf = tfidf_vect.fit_transform(onet_corpus)

# Vectorizing Titles and Descriptions Separately

sample_title = [unquote(str(title)) for title in sample.tj_title]
sample_title = [re.sub('\+', ' ', title) for title in sample_title]
sample_desc = sample.tj_desc

sample_tfidf_title = tfidf_vect.transform(sample_title)
sample_tfidf_desc = tfidf_vect.transform(sample_desc)

# Transforming the Sample

# Calculating Cosine Similarity with Different Weights

wl_title = 0.6
we_title = 1
wl_desc = 1-wl_title
we_desc = 1

sample_title = linear_kernel(sample_tfidf_title, onet_tfidf)
sample_desc = linear_kernel(sample_tfidf_desc, onet_tfidf)
sample_comb = pd.DataFrame(
    data=(sample_title**we_title)*wl_title + (sample_desc**we_desc)*wl_desc,
    columns=onet_data.onet_code,#onet_data.onet_family.unique(),
    index=sample.tj_code)

sample_comb.to_csv(folder_path+'data/outputs/sample_comb.csv', index=False)

del wl_title, we_title, wl_desc, we_desc

# Matching ONET Categories to Job Postings

matches = pd.DataFrame(index=sample.tj_code)

for job in sample.tj_code:
  code = sample_comb.loc[job, sample_comb.columns].idxmax() #.str.startswith(family)].idxmax()
  family = code[0:2]
  matches.loc[job, 'onet_code'] = code
  matches.loc[job, 'onet_family'] = family

matches = matches.reset_index()
matches = matches[matches['onet_family'].notna()]

matches = pd.merge(
    left=matches,
    right=sample[['tj_code', 'tj_title', 'tj_desc', 'tags', 'lockdown_status', 'wfh_status']],
    on='tj_code')
matches = pd.merge(
    left=matches,
    right=onet_data[['onet_code', 'onet_title', 'onet_desc']],
    on='onet_code')
matches.tj_title = [unquote(str(title)) for title in matches.tj_title]
matches.tj_title = [re.sub('\+', ' ', title) for title in matches.tj_title]

matches_title = matches.groupby('onet_family').size()
matches = matches.set_index('tj_code', drop=False)

# Evaluating Matches

for job in matches.tj_code:
  tags = matches.at[job,'tags']
  tag_families = [int(str(tag)[0:2]) for tag in tags if pd.notnull(tag)]
  tag_families = (tag_families + [None]*10)[:10]
  onet_family = matches.at[job,'onet_family']

  matches.loc[job, 'tag_families'] = [[tag_families]]
  matches.loc[job, 'match_value'] = [int(onet_family) in tag_families]

matches.groupby(['wfh_status', 'lockdown_status']).size()

matches.to_csv(folder_path+'data/outputs/sample_matches.csv', index=False)