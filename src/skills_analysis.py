# import the relevant packages
import spacy
from pprint import pprint
import pandas as pd
import nltk
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Download stopwords and spacy model
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def sent_to_words(sentences):
    """Lowercases and converts each sentence into a list of words"""
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(sentence))

def remove_stopwords(texts, stop_words=nltk.corpus.stopwords.words('english')):
    "Removes stopwords using a list of stopwords provided"
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(tokens):
    """build the bi-gram models"""
    
    bigram = gensim.models.Phrases(tokens, min_count=3, threshold=10)
    
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    
    return [bigram_mod[doc] for doc in tokens]

def lemmatization(tokens, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """Do lemmatization keeping only noun, adj, vb, adv"""
    texts_out = []
    for sent in tokens:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def prep_data(df):
    """tokenize, remove stopwords, make bigrams and lemmatize the corpus"""

    tokens = list(sent_to_words(list(df["clean_text"])))
    
    tokens_nostops = remove_stopwords(tokens)
    
    tokens_bigrams = make_bigrams(tokens_nostops)

    tokens_lemmatized = lemmatization(tokens_bigrams)
    
    return tokens_lemmatized
    
def build_topic_models(tokens):
    """train a topic models"""
    #Create data formats necessary to build LDA topic models with gensim
    
    #Create dictionary
    id2word = corpora.Dictionary(tokens)

    # Term Document Frequency from dictionary
    corpus = [id2word.doc2bow(token) for token in tokens]
    
    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=4, 
                                           random_state=100,
                                           update_every=10,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    
    return lda_model
    
    
def main(file_path = "D:/nlp/top_jobs_cs_20_21/part_1/part_1a/p1a.csv"):
    """reads a csv, prepares the data, builds the topic models and prints topic outputs"""
    
    df = pd.read_csv(file_path)
    
    df = df[df["clean_accuracy"]>0]
    
    df = df.sample(200)
    
    tokens_lemmatized = prep_data(df)
    
    lda_model = build_topic_models(tokens_lemmatized)
    
    pprint(lda_model.print_topics())
    
if __name__ == "__main__":
    main()


