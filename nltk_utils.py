import numpy as np
import nltk
#nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):
    """
    spilt sentence into array of words/tokens
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """ stemming convert words into their root words """

    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence,words):
    """turn bag of words array
    stem each word"""
    sentence_words = [stem(word) for word in tokenized_sentence]
    #initialize bag 0 with for each word
    bag = np.zeros(len(words),dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag