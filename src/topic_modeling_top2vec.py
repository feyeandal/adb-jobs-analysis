import numpy as np
import pandas as pd
import re
import yaml
from top2vec import Top2Vec

# Reading the Evaluation Corpus

def read_sample_data(ocr_output_path):
    '''Reads and returns the OCR dataset in the form of a Pandas DataFrame.'''

    sample = pd.read_csv(ocr_output_path)

    return sample

# Topic Modeling Using Top2Vec

def model_topics(df_column, embedding_model, wordclouds_path, topic_model_path):
    '''Executes topic modelling for the dataset.'''

    model = Top2Vec(df_column.values, embedding_model=embedding_model)

    num_topics = model.get_num_topics()

    with open(wordclouds_path, 'w') as fp:
        for i in range (num_topics):
            print(model.topic_words[i])
            model.generate_topic_wordcloud(i)
            fp.write("%s\n" % model.topic_words[i])


    model.save(topic_model_path)   
    return model

def main(ocr_output_path, wordclouds_path, text_column_name, embedding_model, topic_model_path):
    sample = read_sample_data(ocr_output_path)

    model = model_topics(sample[text_column_name].astype(str), embedding_model, wordclouds_path, topic_model_path)

    num_topics = model.get_num_topics()

    for i in range (num_topics):
        print(model.topic_words[i])
        model.generate_topic_wordcloud(i)
