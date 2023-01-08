import matplotlib.image  as mpimg
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sn
import yaml
import logging

# Reading the Evaluation Corpus

def compose_data_sample(matches_path, tags_path):
    '''Composes the data sample by combining corpus of Topjobs data and predicted ONET category matchings with manually annotated tags'''

    # Reading the original Topjobs dataset
    matches = pd.read_csv(matches_path)

    print(matches.onet_family.unique())

    # Reading the manually annotated tags for the data sample
    sample_tags = pd.read_csv(tags_path)

    # Extracting TopJobs vacancy ID codes from image filenames
    # Note: This assumes that the numbers in the filename corresponds to the id
    sample_tags['tj_code'] = sample_tags.file_name.replace(to_replace='[^0-9]', value='', regex=True).astype(int)
    
    # Composing a single dataset by appending manually annotated tags to the Topjobs dataset    
    # Let's check whether the dataset being processed have any images that we hand annotated
    sample_matches_overlap = pd.merge(left=sample_tags, right=matches, how='inner', on='tj_code')

    if len(sample_matches_overlap) == 0:
        logging.warning('There is no overlapping images between the predicted ones and hand tagged ones. Evaluation aborted!')

        return None

    ## The downstream steps need the tags to stored in a list 
    # Columns that contain a tag
    tag_columns = [x for x in sample_matches_overlap.columns if x.startswith('tag_')]
    sample_matches_overlap['tags'] = sample_matches_overlap[tag_columns].values.tolist()
    sample_matches_overlap.drop(columns=tag_columns, inplace=True)

    # Freeing the memory of unused variables
    del sample_tags, matches

    return sample_matches_overlap

# Evaluating Matches
def evaluate_matches(matches):
    '''Evaluates the performance of the tf-idf vectorizer by generating a confusion matrix between manually anotated ONET categories and those annotated via tf-idf.'''

    matches.set_index('tj_code', inplace=True)

    for job, row in matches.iterrows():

        logging.debug(f'Processing {job}')

        tag_families = [int(str(tag)[0:2]) for tag in row['tags'] if pd.notnull(tag)]
        
        tag_families = (tag_families + [None]*10)[:10]

        matches.loc[job, 'tag_families'] = [[tag_families]]
        matches.loc[job, 'first_tag_family'] = str(tag_families[0])

        # Question: What does this line do? It's already an integer?  
        matches.loc[job, 'match_value'] = [int(x) for x in tag_families if x is not None]

    matches.groupby(['wfh_status', 'lockdown_status']).size()

    # Generating the confusion matrix
    confusion_matrix = pd.crosstab(matches.first_tag_family, matches.onet_family, rownames=['Actual'], colnames=['Predicted'])

    # Generating the heatmap for the confusion matrix
    sn.heatmap(confusion_matrix, annot=True)
    plt.show()

    return matches, confusion_matrix


def main(matches_path, tags_path):
    data_sample_for_eval = compose_data_sample(matches_path, tags_path)

    if data_sample_for_eval is not None:
        matches, confusion_matrix = evaluate_matches(data_sample_for_eval)
    else:
        matches, confusion_matrix = None, None

    return (matches, confusion_matrix)
