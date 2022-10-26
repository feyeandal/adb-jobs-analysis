import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sn
import yaml

# Reading the Evaluation Corpus

def compose_data_sample(matches_path, tags_path):
    '''Composes the data sample by combining corpus of Topjobs data and predicted ONET category matchings with manually annotated tags'''

    # Reading the original Topjobs dataset
    matches = pd.read_csv(matches_path)

    # Reading the manually annotated tags for the data sample
    sample_tags = pd.read_csv(tags_path)

    # Extracting TopJobs vacancy ID codes from image filenames
    sample_tags['tj_code'] = sample_tags.file_name.replace(to_replace='[^0-9]', value='', regex=True).astype(int)
    
    # Composing a single dataset by appending manually annotated tags to the Topjobs dataset
    sample = pd.merge(left=sample_tags, right=matches, how='left', on='tj_code')

    sample['tags'] = sample[['tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8', 'tag_9', 'tag_10']].values.tolist()
    sample = sample.drop(columns=['tag_1', 'tag_2', 'tag_3', 'tag_4', 'tag_5', 'tag_6', 'tag_7', 'tag_8', 'tag_9', 'tag_10'])

    del sample_tags, matches

    return sample

# Evaluating Matches

def evaluate_matches(matches, matches_path):
    '''Evaluates the performance of the tf-idf vectorizer by generating a confusion matrix between manually anotated ONET categories and those annotated via tf-idf.'''

    for job in matches.tj_code:
        tags = matches.at[job,'tags']
        tag_families = [int(str(tag)[0:2]) for tag in tags if pd.notnull(tag)]
        tag_families = (tag_families + [None]*10)[:10]
        onet_family = matches.at[job,'onet_family']

        matches.loc[job, 'tag_families'] = [[tag_families]]
        matches.loc[job, 'first_tag_family'] = str(matches.loc[job, 'tag_families'][0][0])
        matches.loc[job, 'match_value'] = [int(onet_family) in tag_families]

    matches.groupby(['wfh_status', 'lockdown_status']).size()

    # Generating the confusion matrix
    confusion_matrix = pd.crosstab(matches.first_tag_family, matches.onet_family, rownames=['Actual'], colnames=['Predicted'])
    print (confusion_matrix)

    # Generating the heatmap for the confusion matrix
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()

    return matches, confusion_matrix

def main(matches_path, tags_path):
    matches = compose_data_sample(matches_path, tags_path)

    matches, confusion_matrix = evaluate_matches(matches, matches_path)

    return (matches,confusion_matrix)
