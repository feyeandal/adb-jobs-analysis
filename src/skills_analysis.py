import spacy
from pprint import pprint

import pandas as pd
import nltk
nltk.download('stopwords')

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


file_path = "E:/ADB_Project/code/data/pipeline_sample.csv"
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(sentence))

def remove_stopwords(texts):
    stop_words = nltk.corpus.stopwords.words('english')
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(tokens, bigram):
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    return [bigram_mod[doc] for doc in tokens]

def lemmatization(tokens, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in tokens:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def main():

    df = pd.read_csv(file_path)

    tokens = list(sent_to_words(list(df["clean"])))
    tokens_nostops = remove_stopwords(tokens)

    bigrams = gensim.models.Phrases(tokens, min_count=3, threshold=10)
    tokens_bigrams = make_bigrams(tokens_nostops, bigrams)

    tokens_lemmatized = lemmatization(tokens_bigrams)

# Create Dictionary
id2word = corpora.Dictionary(tokens_lemmatized)

# Create Corpus
texts = tokens_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5, 
                                           random_state=100,
                                           update_every=10,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

#Print the Keyword in the 20 topics
pprint(lda_model.print_topics())


