#!/usr/bin/env python

# softtfidf.py
#
# Soft TF-IDF String Comparison Algorithm
# 
# Copyright 2015 Massachusetts Institute of Technology, Lincoln Laboratory
# version 0.1
#
# author: Charlie Dagli
# dagli@ll.mit.edu
#
# Original logic written by @drangons for the entity_resolution_spark repository:
# https://github.com/drangons/entity_resolution_spark
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


#Imports
import os
from collections import namedtuple
import logging
import math

from  sklearn.feature_extraction.text import CountVectorizer
from  sklearn.feature_extraction.text import TfidfTransformer
from  sklearn.feature_extraction.text import TfidfVectorizer

import jellyfish as jf


class Softtfidf:
    """
    This module implements the soft tf-idf algorithm described in:
        A Comparison of String Distance Metrics for Name-Matching Tasks
        Cohen et al., IJCAI 2003
    """

    # Logging
    LOG_LEVEL = logging.DEBUG
    logging.basicConfig(level=LOG_LEVEL,
                                format='%(asctime)s %(levelname)-8s %(message)s',
                                                    datefmt='%a, %d %b %Y %H:%M:%S')
    logger = logging.getLogger(__name__)


    def __init__(self,threshold=0.6, idf_model=None):
        '''
        Constructor 
        '''
        self.THRESHOLD = threshold

        if idf_model == None:
            self.LOG_IDF = None
            self.CORPUS_VOCAB = None
            self.OOV_IDF_VAL = None
        else:
            self.LOG_IDF = idf_model['idf']
            self.CORPUS_VOCAB = idf_model['corpus_vocab']
            self.OOV_IDF_VAL = idf_model['oov_idf_val']


    def set_model(self,idf_model):
        '''
        Set softtfidf matcher's model parameters
        '''
        # Set (or compute) IDF and corresponding vocabulary
        self.LOG_IDF = idf_model['idf']
        self.CORPUS_VOCAB = idf_model['corpus_vocab']
        self.OOV_IDF_VAL = idf_model['oov_idf_val']


    def set_threshold(self,threshold=0.6):
        '''
        Set threshold
        '''
        self.THRESHOLD = threshold


    def compute_VwS(self,s):
        '''
        Compute V(w,S) as defined by Cohen et al.'s IJCAI03 paper
        '''
        # Get term-frequency vectors and vocab list for string
        #cv = CountVectorizer(min_df = 0.0, token_pattern=)
        #tf = cv.fit_transform([s,s]); tf = tf.tocsr(); tf = tf[0,:]
        cv = CountVectorizer(min_df = 0.0, token_pattern=u'(?u)\\b\\w+\\b')
        #self.logger.info(s)
        tf = cv.fit_transform([s]); tf = tf.tocsr()
        vocab = cv.vocabulary_

        # Compute V(w,S) for string
        vprime_ws = dict()
        vprime_ws_norm = 0
        for w in vocab:
            if w in self.CORPUS_VOCAB:
                vprime_ws[w] = math.log(tf[0,vocab[w]]+1)*self.LOG_IDF[self.CORPUS_VOCAB[w]]
            else:
                vprime_ws[w] = math.log(tf[0,vocab[w]]+1)*self.OOV_IDF_VAL #if not in vocab, defauly to OOC_IDF_VAL
            vprime_ws_norm += vprime_ws[w]**2
        vprime_ws_norm = math.sqrt(vprime_ws_norm)

        return (vocab,vprime_ws,vprime_ws_norm)


    def score(self,s,t):
        '''
        Returns the soft tf-idf similarity
        '''

        # Check to see whether a model exists; otherwise default to degenerate solution
        if (self.LOG_IDF is None) | (self.CORPUS_VOCAB is None) | (self.OOV_IDF_VAL is None):
            self.logger.info("Either (or both) IDF or corpus vocabulary parameters not given " 
                                +"Defaulting to degenerate mode where corpus consists only of the "
                                +"two strings given as input.");
            self.compute_query_idf([s,t])

        # Get V(w,S) and V(w,T) (along with vocab lists for s and t) 
        try: 
            (s_vocab,vprime_ws,vprime_ws_norm) = self.compute_VwS(s)
            (t_vocab,vprime_wt,vprime_wt_norm) = self.compute_VwS(t)
        except ValueError:
            self.logger.info("string got stop-listed; most likely b/c" , \
                    "it is of length 1, with the only character being a ", \
                    "non-normalized punctuation mark. (i.e. '.')")
            sim = 0.0
            return sim

        #compute D(w,T) for all w
        max_vT = dict()
        jw_sims = dict()
        for w in s_vocab:
            max_vT[w] = dict(); max_vT[w]['score'] = 0.0; max_vT[w]['max_v'] = '';
            jw_sims[w] = dict()
            for v in t_vocab:
                dist = jf.jaro_winkler(w,v)
                jw_sims[w][v] = dist
                if (dist >= max_vT[w]['score']):
                    max_vT[w]['score'] = dist
                    max_vT[w]['max_v'] = v
        self.logger.debug("max_vT: {0}".format(max_vT))

        # compute soft tf-idf sim
        sim = 0.0
        self.logger.debug(s_vocab)
        for w in s_vocab:
            for v in t_vocab:
                if (jw_sims[w][v] >= self.THRESHOLD):
                    inner_sum = (vprime_ws[w]/vprime_ws_norm)*(vprime_wt[max_vT[w]['max_v']]/vprime_wt_norm)*max_vT[w]['score']
                    self.logger.debug(u"(w,vprime_ws[w],vprime_ws_norm): ({0},{1},{2})".format(w,vprime_ws[w],vprime_ws_norm))
                    self.logger.debug(u"(max_vT[w]['max_v'],vprime_wt[max_vT['max_v'],vprime_wt_norm): ({0},{1},{2})".format(max_vT[w]['max_v'],vprime_wt[max_vT[w]['max_v']],vprime_wt_norm))
                    self.logger.debug(u"(max_vT[w]['score']): ({0})".format(max_vT[w]['score']))
                    self.logger.debug(u"(w,v,inner_sum): ({0},{1},{2})".format(w,v,inner_sum))
                    sim += inner_sum
                    break

        self.logger.debug("Soft TF-IDF Similarity: {0}".format(sim))

        return sim


    def compute_query_idf(self,corpus):
        '''
        Compute IDF from s and t in case you have no externally computed IDF to use 
        '''
        cv = CountVectorizer(min_df = 0.0)
        cv.fit_transform(corpus)
        self.logger.debug(cv.vocabulary_)
        freq_term_matrix = cv.transform(corpus)
        tfidf = TfidfTransformer(norm="l2")
        tfidf.fit(freq_term_matrix)
        log_idf = tfidf.idf_
        self.LOG_IDF = log_idf
        self.CORPUS_VOCAB = cv.vocabulary_

