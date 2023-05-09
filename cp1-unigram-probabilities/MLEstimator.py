'''
Summary
-------
Defines a maximum likelihood estimator for unigrams

MLEstimator supports a common API for unigram probability estimators:
* fit
* predict_proba
* score

Resources
---------
See COMP 136 course website for the complete problem description and all math details
'''

import numpy as np
from Vocabulary import Vocabulary
import time


class MLEstimator():
    """
    Maximum Likelihood Estimator for unigram probabilities

    To avoid pathologies with unseen words,
    we perform some ad-hoc smoothing so that this estimator:
    * 1) is a valid PMF over the vocabulary
    * 2) does not give zero probability to any word

    Attributes
    ----------
    vocab : Vocabulary object
    unseen_proba : float between 0.0 and 1.0
            Probability mass allowed for all unseen words

    Examples
    --------
    ## Note: Will NOT pass with starter code
    >>> word_list = ['dinosaur', 'trex', 'dinosaur', 'stegosaurus']
    >>> mle = MLEstimator(Vocabulary(word_list), unseen_proba=0.1)
    >>> mle.fit(word_list)
    >>> mle.predict_proba('dinosaur')
    0.45

    >>> mle.predict_proba('never_seen-before')
    Traceback (most recent call last):
    ...
    KeyError: 'Word never_seen-before not in the vocabulary'
    """

    def __init__(self, vocab, unseen_proba=0.000001): 
        self.vocab = vocab
        self.unseen_proba = unseen_proba

        # State that is adjusted by calls to 'fit'
        self.total_count = 0
        self.unseen_count = 0
        self.count_V = None


    def fit(self, word_list):
        ''' Fit this estimator to provided training data

        Args
        ----
        word_list : list of str
                Each entry is a word that can be looked up in the vocabulary

        Returns
        -------
        None. Internal attributes updated.

        Post Condition
        --------------
        Attributes will updated based on provided word list
        * The 1D array count_V is set to the count of each vocabulary word
        * The integer total_count is set to the total length of the word list
        '''
        self.count_V = np.zeros(self.vocab.size)
        for word in self.vocab.vocab_dict:
            if word_list.count(word) > 0:
                self.count_V[self.vocab.get_word_id(word)] += word_list.count(word)
            else:
                self.unseen_count += 1


        self.total_count = len(word_list)


    def predict_proba(self, word):
        ''' Predict probability of a given unigram under this model

        Assumes this word is in the vocabulary

        Args
        ----
        word : string
                Known word that can be looked up in the vocabulary

        Returns
        -------
        proba : float between 0 and 1

        Raises
        ------
        KeyError if the provided word is not in the vocabulary
        '''
        # print("predicting probability")
        word_in_dict = self.vocab.get_word_id(word)

        if self.count_V[word_in_dict] > 0:
            return (1 - self.unseen_proba) * self.count_V[word_in_dict]/self.total_count
        else:
            return self.unseen_proba / self.unseen_count

    def score(self, word_list):
        ''' Compute the average log probability of words in provided list

        Args
        ----
        word_list : list of str
                Each entry is a word that can be looked up in the vocabulary

        Returns
        -------
        avg_log_proba : float between (-np.inf, 0.0)
        '''
        total_log_proba = 0.0
        for word in word_list:
            total_log_proba += np.log(self.predict_proba(word))
        return total_log_proba / len(word_list)
