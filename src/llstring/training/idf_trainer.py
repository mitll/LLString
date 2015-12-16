#!/usr/bin/env python

# idf_trainer.py
#
# Class to learn IDF weighting from training data
# 
# Copyright 2015 Massachusetts Institute of Technology, Lincoln Laboratory
# version 0.1
#
# author: Charlie K. Dagli
# dagli@ll.mit.edu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import cPickle as pickle

from  sklearn.feature_extraction.text import CountVectorizer
from  sklearn.feature_extraction.text import TfidfTransformer


class IDFTrainer:
    '''
    Class to learn IDF weighting from training data
    '''

    # Logging
    LOG_LEVEL = logging.INFO
    logging.basicConfig(level=LOG_LEVEL,
                                format='%(asctime)s %(levelname)-8s %(message)s',
                                                    datefmt='%a, %d %b %Y %H:%M:%S')
    logger = logging.getLogger(__name__)
    

    def __init__(self,min_df=2,norm="l2"):
        '''
        Constructor
        '''
        self.cv = CountVectorizer(min_df=min_df)
        self.tfidf = TfidfTransformer(norm)

        self.LOG_IDF = None
        self.CORPUS_VOCAB = None
        self.OOV_IDF_VAL = 0 #min idf value to assign for out-of-vocabulary terms

        self.IDF_MODEL = dict()


    def compute_idf(self,corpus):
        '''
        Compute IDF using corpus.
        Per sklearn conventions, "corpus" can be either a:
            file: a file object for a file containing content (newline separated) 
            content: a iterable containing all the data in memory (i.e. a list)
            filename: list of filenames of documents in which content is contained
        '''
        self.cv.fit_transform(corpus)
        self.logger.debug(self.cv.vocabulary_)
        self.CORPUS_VOCAB = self.cv.vocabulary_
        self.logger.debug(self.CORPUS_VOCAB)

        # if corpus is file object, seek back to beginning of file...
        if isinstance(corpus,file):
            corpus.seek(0)

        freq_term_matrix = self.cv.transform(corpus)
        self.tfidf.fit(freq_term_matrix)

        self.LOG_IDF = self.tfidf.idf_
        self.N = freq_term_matrix.shape[0] #num of "docs" processed
        
        if isinstance(corpus,file):
            corpus.close()

        # Compute OOV_IDF_VAL: min idf value to assign for out-of-vocabulary terms
        nt=1; self.OOV_IDF_VAL = math.log(self.N/(nt+1))+1

        # collect model components
        self.IDF_MODEL['idf'] = self.LOG_IDF
        self.IDF_MODEL['corpus_vocab'] = self.LOG_IDF
        self.IDF_MODEL['oov_idf_val'] = self.OOV_IDF_VAL


    def save_model(self,fnameout):
        '''
        Save-out learned IDF dictionary and associated metadata (e.g. self.IDF_MODEL)
        '''
        self.logger("saving IDF model to {0}".format(fnameout))
        pickle.dump(self.IDF_MODEL,open(fnameout,"wb"))

