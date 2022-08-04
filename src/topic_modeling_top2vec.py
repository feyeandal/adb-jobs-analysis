import numpy as np
import pandas as pd
from top2vec import Top2Vec
import re

folder_path = '/content/drive/MyDrive/LIRNEasia/ADB Project/'

# Reading the Evaluation Corpus

def read_sample_data(data_path, tags_path):
    data_full = pd.read_excel(data_path)

    image_path = folder_path+'data/sample/'
    sample_tags = pd.read_csv(tags_path)

    sample_tags['image_drive_url'] = image_path + sample_tags['file_name']
    sample_tags['job_code'] = sample_tags.file_name.replace(
        to_replace='[^0-9]', value='', regex=True).astype(int)

    sample = pd.merge(left=sample_tags, right=data_full, how='left', on='job_code')
    sample['tags'] = sample[['tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8', 'tag_9', 'tag_10']].values.tolist()
    sample = sample.drop(columns=['tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8', 'tag_9', 'tag_10'])

    del sample_tags, data_full

    return sample

# Cleaning Text

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

# Getting the OCR Outputs

def get_ocr_output(ocr_path, sample):
    sample.set_index('job_code', drop=False)

    sample_ocr_output = pd.read_csv(ocr_path)
    sample_ocr_output = sample_ocr_output.drop(columns=['file_name', 'image_drive_url'])
    sample = pd.merge(left=sample, right=sample_ocr_output, how='left', on='job_code')

    sample['tj_desc'] = [clean_text(text) for text in sample.ocr_output]
    sample['tj_desc'] = sample['tj_desc'].str.lower()

    return sample

# Renaming Relevant Columns, Adding Status Columns, and Removing Unnecessary Columns

def prepare_sample(sample):

    sample = sample.rename(columns={'job_code': 'tj_code', 'job_title': 'tj_title', })
    sample = sample.drop(columns=['image_drive_url', 'job_description', 'remark', 'functional_area', 'expiry_date', 'image_string', 'image_source', 'image_code', 'image_url', 'start_date'])
    
    return sample

# Topic Modeling Using Top2Vec

def model_topics(df_column, embedding_model):
    model = Top2Vec(df_column.values, embedding_model=embedding_model)

    num_topics = model.get_num_topics()

    for i in range (num_topics):
        print(model.topic_words[i])
        model.generate_topic_wordcloud(i)
    
    return model

data_path = folder_path+'data/data_full.xlsx'
tags_path = folder_path+'data/cs_sample_tags.csv'
ocr_path = folder_path+'data/outputs/cs_sample_ocr_output.csv'
embedding_model = 'universal-sentence-encoder'

sample = read_sample_data(data_path, tags_path)
sample = get_ocr_output(ocr_path, sample)
sample = prepare_sample(sample)

model = model_topics(sample['tj_desc'], embedding_model)
