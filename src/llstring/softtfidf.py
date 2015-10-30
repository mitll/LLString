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
    LOG_LEVEL = logging.INFO
    logging.basicConfig(level=LOG_LEVEL,
                                format='%(asctime)s %(levelname)-8s %(message)s',
                                                    datefmt='%a, %d %b %Y %H:%M:%S')
    logger = logging.getLogger(__name__)


    def __init__(self,idf_model=None,threshold=0.6):
        '''
        Constructor 
        '''

        # Set (or compute) IDF and corresponding vocabulary
        if idf_model is not None:
            self.LOG_IDF = idf_model['idf']
            self.CORPUS_VOCAB = idf_model['corpus_vocab']
            self.OOV_IDF_VAL = idf_model['oov_idf_val']
        else:
            self.logger.info("Either (or both) IDF or corpus vocabulary parameters not given. 
                                Defaulting to corpus formed by input strings");
            self.CORPUS.append(s)
            self.CORPUS.append(t)

            self.compute_query_idf()

        # Set threshold
        self.THRESHOLD = threshold
            

    def compute_VwS(self,s):
        '''
        Compute V(w,S) as defined by Cohen et al.'s IJCAI03 paper
        '''
        # Get term-frequency vectors and vocab list for string
        cv = CountVectorizer(min_df = 0.0)
        tf = cv.fit_transform([s,s]); tf = tf.tocsr(); tf = tf[0,:]
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

        # Get V(w,S) and V(w,T) (along with vocab lists for s and t) 
        (s_vocab,vprime_ws,vprime_ws_norm) = self.compute_VwS(s)
        (t_vocab,vprime_wt,vprime_wt_norm) = self.compute_VwS(t)

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
        for w in s_vocab:
            for v in t_vocab:
                self.logger.debug("(w,v,dist): ({0},{1},{2})".format(w,v,dist))
                if (jw_sims[w][v] >= self.THRESHOLD):
                    inner_sum = (vprime_ws[w]/vprime_ws_norm)*(vprime_wt[max_vT[w]['max_v']]/vprime_wt_norm)*max_vT[w]['score']
                    sim += inner_sum
                    break

        self.logger.debug("Soft TF-IDF Similarity: {0}".format(sim))

        return sim


    def compute_query_idf(self):
        '''
        Compute IDF from s and t in case you have no externally computed IDF to use 
        '''
        cv = CountVectorizer(min_df = 0.0)
        cv.fit_transform(self.CORPUS)
        self.logger.debug(cv.vocabulary_)
        freq_term_matrix = cv.transform(self.CORPUS)
        tfidf = TfidfTransformer(norm="l2")
        tfidf.fit(freq_term_matrix)
        log_idf = tfidf.idf_
        self.LOG_IDF = log_idf
        self.CORPUS_VOCAB = cv.vocabulary_

