#####################################################################
# Text Helper
# 
# To process text data
# 
#####################################################################

__author__ = "Kornraphop Kawintiranon"
__email__ = "kornraphop.k@gmail.com"

import string
import tqdm
import re
import concurrent.futures
import multiprocessing
from nltk.corpus import stopwords
from nltk.tokenize.destructive import NLTKWordTokenizer

def parallel_tokenize(corpus, tokenizer=None, n_jobs=-1):
    if tokenizer == None:
        tokenizer = NLTKWordTokenizer()
    if n_jobs < 0:
        n_jobs = multiprocessing.cpu_count() - 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        corpus_tokenized = list(
            tqdm.tqdm(executor.map(tokenizer.tokenize, corpus, chunksize=200), total=len(corpus), desc='Tokenizing')
        )
    return corpus_tokenized

def remove_stopwords(corpus, language='english'):
    stop_words = set(stopwords.words(language))
    processed_corpus = []
    for words in corpus:
        words = [w for w in words if not w in stop_words]
        processed_corpus.append(words)
    return processed_corpus

def remove_punctuations(corpus):
    punctuations = string.punctuation
    processed_corpus = []
    for words in corpus:
        words = [w for w in words if not w in punctuations]
        processed_corpus.append(words)
    return processed_corpus
    
def decontract(corpus):
    processed_corpus = []
    for phrase in tqdm.tqdm(corpus, desc="Decontracting"):
        phrase = re.sub(r"’", "\'", phrase)

        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)

        processed_corpus.append(phrase)
    return processed_corpus

def get_word_counts(corpus):
    # Initializing Dictionary
    d = {}

    # Counting number of times each word comes up in list of words (in dictionary)
    for words in tqdm.tqdm(corpus, desc="Word Counting"):
        for w in words:
            d[w] = d.get(w, 0) + 1
    return d

if __name__ == "__main__":
    """
    For testing
    """
    
    texts = [
        "I love NLP.",
        "I also really like machine learning. I would like to work on this field."
    ]

    tokenized_texts = parallel_tokenize(texts)
    for tokens in tokenized_texts:
        print(tokens)
    processed_texts = remove_stopwords(tokenized_texts)
    for tokens in processed_texts:
        print(tokens)
