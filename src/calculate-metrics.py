import pandas as pd
import numpy as np
import re

from urllib.parse import unquote

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def nltk_tokenizer(text):
    tokens = [word for word in word_tokenize(text)]
    stems = [PorterStemmer().stem(word) for word in tokens]
    return stems


# Read in the data ------------------------------------------------------------
tj_meta = pd.read_excel('data/metadata.xlsx') \
    .rename(columns={'job_code': 'tj_code', 'job_title': 'tj_title'}) \
    .drop(columns=['job_description', 'remark'])
tj_desc = pd.read_csv('data/ocr_sample_output.csv') \
    .rename(columns={'id': 'tj_code', 'cleaned_text': 'tj_desc'}) \
    .drop(columns=['Unnamed: 0', 'ocr_output', 'plain_accuracy'])
tj_desc.tj_code = tj_desc.tj_code.replace(
    to_replace='[^0-9]', value='', regex=True).astype(int)
tj_data = pd.merge(left=tj_meta, right=tj_desc, how='right', on='tj_code')

del tj_meta, tj_desc

onet_occ = pd.read_csv('data/onet_occ_titles.txt', sep='\t')
onet_alt = pd.read_csv('data/onet_alt_titles.txt', sep='\t') \
    .groupby(by='onet_code') \
    .agg({'onet_title_alt': lambda x: x.astype(object)}) \
    .reset_index()
onet_tech = pd.read_csv('data/onet_tech_skills.txt', sep='\t') \
    .groupby(by='onet_code') \
    .agg({'onet_tech': lambda x: x.astype(object)}) \
    .reset_index()
onet_data = pd.merge(left=onet_occ, right=onet_alt, how='left', on='onet_code')
onet_data = pd.merge(
    left=onet_data, right=onet_tech, how='left', on='onet_code') \
    .fillna('')

del onet_occ, onet_alt, onet_tech

# Job post count disaggregated by sector --------------------------------------
tj_data['lockdown'] = tj_data.start_date >= '2020-01-01'
tj_data.groupby(by=['functional_area', 'lockdown']).size()

# Job post count disaggregated by job title -----------------------------------
# Create the relevant corpuses
onet_corpus = onet_data.onet_title + ' ' + \
    [' '.join(titles) for titles in onet_data.onet_title_alt] + \
    [' '.join(techs) for techs in onet_data.onet_tech] + \
    ' ' + onet_data.onet_desc
tj_corpus_title = [unquote(str(title)) for title in tj_data.tj_title]
tj_corpus_title = [re.sub('\+', ' ', title) for title in tj_corpus_title]
tj_corpus_desc = tj_data.tj_desc

# Fit the tf-idf vectorizer on the reference corpus
tfidf_vect = TfidfVectorizer(tokenizer=nltk_tokenizer, stop_words='english')
onet_tfidf = tfidf_vect.fit_transform(onet_corpus)

# Vectorize the titles and descriptions separately
tj_tfidf_title = tfidf_vect.transform(tj_corpus_title)
tj_tfidf_desc = tfidf_vect.transform(tj_corpus_desc)

# Calculate the cosing similarity with different weights
wl_title = 0.5
we_title = 1
wl_desc = 1-wl_title
we_desc = 1

cs_title = linear_kernel(tj_tfidf_title, onet_tfidf)
cs_desc = linear_kernel(tj_tfidf_desc, onet_tfidf)
cs_comb = pd.DataFrame(
    data=(cs_title**we_title)*wl_title + (cs_desc**we_desc)*wl_desc,
    columns=onet_data.onet_code,
    index=tj_data.tj_code)

del wl_title, we_title, wl_desc, we_desc

# Find the job family with the highest average score
matches_family = pd.DataFrame()
for family in set([col_name[0:2] for col_name in onet_data.onet_code]):
    matches_family[family] = np.mean(cs_comb.filter(
        regex=r'^' + re.escape(family)), axis=1)

matches_family['match'] = matches_family.idxmax(axis=1)

del family

# Find the job with the highest score in that family
matches = pd.DataFrame(index=tj_data.tj_code)
for job in tj_data.tj_code:
    family = matches_family.match[job]
    code = cs_comb.loc[job, cs_comb.columns.str.startswith(family)].idxmax()
    matches.loc[job, 'onet_code'] = code

matches = matches.reset_index()

del family, job, code

# Store the matches in an easy-to-read format
matches = pd.merge(
    left=matches,
    right=tj_data[['tj_code', 'tj_title', 'tj_desc']],
    on='tj_code')
matches = pd.merge(
    left=matches,
    right=onet_data[['onet_code', 'onet_title', 'onet_desc']],
    on='onet_code')
matches.tj_title = [unquote(str(title)) for title in matches.tj_title]
matches.tj_title = [re.sub('\+', ' ', title) for title in matches.tj_title]

matches_family.groupby('match').size()
matches_title = matches.groupby('onet_title').size()

# Things to try with the TF-IDF approach
# 1. Add/Remove alt titles/techs/skills to/from the ONET corpus
# 2. Adjust linear weights of the title and description (wl_title/wl_desc)
# 3. Adjust exponential weights of the title and description (we_title/we_desc)

# Different approaches to try
# 1. Identify keywords in TJ data and feed it directly into the ONET search API
# 2. Take titles/keywords from ONET data and match in TJ data to create tags
# 3. Direct title matching followed by TF-IDF approach to break ties
# 4. More sophisticated word embedding techniques