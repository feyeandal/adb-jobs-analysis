import numpy as np
import pandas as pd
import math
import nltk
import gensim

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(sentence))
        
def uniqueWordList(token_list):
    flat_list = [item for sublist in token_list for item in sublist]
    return list(set(flat_list))

def computeIDF(token_list, doclist):
    
    unique_wordlist = uniqueWordList(token_list)
    
    IDF = dict.fromkeys(unique_wordlist, 0)
    for word in unique_wordlist:
        num_doc = 0
        for doc in doclist:
            if word in doc:
                num_doc += 1
        IDF[word] = math.log(len(doclist)/num_doc)
    return IDF

def custom_stopwords(idf):
    
    # get the dictionary values into an array
    value_array = np.array(list(idf.values())) 

    # set the upper and lower boundaries
    upper = np.percentile(value_array, 95)
    lower = np.percentile(value_array, 5)
    
    custom_stop_words = []
    for k, v in idf.items():
        if v > upper or v < lower:
            custom_stop_words.append(k)
    return custom_stop_words

def remove_stopwords(texts, csw_list=None, csw_stopwords=False):
    
    #append the relevant custom stopword list to the standard stop word list
    if csw_stopwords == True:
        stop_words = stop_words_nltk + csw_list
    else:
        stop_words = stop_words_nltk
        
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(tokens, bigram):
    
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    
    return [bigram_mod[doc] for doc in tokens]

def make_trigrams(tokens, trigram, bigram):
    
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    return [trigram_mod[bigram_mod[doc]] for doc in tokens]

def lemmatization(tokens, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in tokens:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

nltk.download('stopwords')

# NLTK Stop words
from nltk.corpus import stopwords
stop_words_nltk = stopwords.words('english')

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])



# Create Dictionary
id2word_20_cs_cf = corpora.Dictionary(tokens_lemmatized_20_cs_cf)

# Create Corpus
texts_20_cs_cf = tokens_lemmatized_20_cs_cf

# Term Document Frequency
corpus_20_cs_cf = [id2word_20_cs_cf.doc2bow(text) for text in texts_20_cs_cf]


# Build LDA model
lda_model_20_cs_cf = gensim.models.ldamodel.LdaModel(corpus=corpus_20_cs_cf,
                                           id2word=id2word_20_cs_cf,
                                           num_topics=25, 
                                           random_state=100,
                                           update_every=10,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)

#Print the Keyword in the 20 topics
pprint(lda_model_20_cs_cf.print_topics())

#Function to alculate coherence score
def get_coherence(model, texts, dictionary, coherence="c_v"):
    
    #coherence score
    coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence = coherence_model.get_c
    oherence()
    
    return coherence

# Calculate coherence score
coherence_score_lda_20_cs_cf = coherence_model_lda_20_cs_cf.get_coherence()
print('\nCoherence Score: ', coherence_score_lda_20_cs_cf)

# LDA Gensim (MALLET) implementation (5-15 Topics) 

from gensim.models.wrappers import LdaMallet
os.environ.update({'MALLET_HOME':r'C:/mallet-2.0.8/'}) 

#You should update this path as per the path of Mallet directory on your system.
new_mallet_path = r'C:/new_mallet/mallet-2.0.8/bin/mallet'

ldamallet_20_cs_cf = gensim.models.wrappers.LdaMallet(new_mallet_path, corpus=corpus_20_cs_cf, optimize_interval=10, num_topics=25, 
                                                     id2word=id2word_20_cs_cf, workers=12)


pprint(ldamallet.print_topics())


