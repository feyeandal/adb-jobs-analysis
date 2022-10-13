# import the relevant packages
import spacy
from pprint import pprint
import pandas as pd
import nltk
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import yaml

# Download stopwords and spa
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
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in tokens:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def prep_data(df):
    """build the bi-gram models"""

    tokens = list(sent_to_words(list(df["clean_text"])))
    
    tokens_nostops = remove_stopwords(tokens)
    
    tokens_bigrams = make_bigrams(tokens_nostops)

    tokens_lemmatized = lemmatization(tokens_bigrams)
    
    return tokens_lemmatized
    
def build_topic_models(tokens_lemmatized):
    """topic models"""
    #Create data formats necessary to build LDA topic models with gensim
    
    #Create dictionary
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
    
    return lda_model
    
    
def main(file_path):
    
    df = pd.read_csv(file_path)
    
    tokens_lemmatized = prep_data(df)
    
    lda_model = build_topic_models(tokens_lemmatized)
    
    #Print the Keyword in the 20 topics
    pprint(lda_model.print_topics())
    
if __name__ == "__main__":
    # Reading config.yaml
    with open("config.yaml", 'r') as stream:
        config_dict = yaml.safe_load(stream)
    
    # Path to the OCR outputs for the Topjobs data sample
    ocr_output_path = config_dict.get("ocr_output_path")

    main(ocr_output_path)

