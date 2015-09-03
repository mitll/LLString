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
import numpy as np
from  sklearn.feature_extraction.text import CountVectorizer
from  sklearn.feature_extraction.text import TfidfTransformer
from  sklearn.feature_extraction.text import TfidfVectorizer
import jellyfish as jf


class Softtfidf:
    """
    This module implements the soft tf-idf algorithm described in paper


    This algorithm is best suited for record matching where the record is generally
    smaller compared to document

    Steps:
        1. Compute the tf.idf score of document corpus
        2. Score method return the soft tf-idf of the query against the record in the
        corpus
    """

    # get projectdir following project dir conventions...
    projectdir = os.path.realpath(__file__).split('src')[0]

    # projectname via project dir conventions...
    projectname = projectdir.split(os.sep)[-2:-1][0]

    #current package name (i.e. containing directory name)
    pkgname = os.path.realpath(__file__).split(os.sep)[-2]

    #class name
    classname = __name__

    # Logging
    LOG_LEVEL = logging.INFO
    logging.basicConfig(level=LOG_LEVEL,
                                format='%(asctime)s %(levelname)-8s %(message)s',
                                                    datefmt='%a, %d %b %Y %H:%M:%S')
    logger = logging.getLogger(__name__)

    THRESHOLD = 0.6

    def compute_idf(self):
        '''
        Scratch
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
        

    def __init__(self):
        self.tfidfvector = TfidfVectorizer(min_df=0, norm='l2', tokenizer=lambda x:x.split(" "))
        self.CORPUS = []

    def buildcorpus(self):
        '''
        Returns sparse vector of tfidf score
        '''
        return self.tfidfvector.fit_transform(self.CORPUS)

    def builddict(self):
        '''
        Returns dictionary of words as key and tfidf score as value
        '''
        matrix = self.buildcorpus()
        vocabulary = self.tfidfvector.vocabulary_
        tfidfdict ={}
        for docId,doc in enumerate(self.CORPUS):
            for word in doc.split(" "):
                tfidfdict[word]=matrix[(docId,vocabulary[word])]
        return tfidfdict

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
            vprime_ws[w] = np.log(tf[0,vocab[w]]+1)*self.LOG_IDF[self.CORPUS_VOCAB[w]]
            vprime_ws_norm += vprime_ws[w]**2
        vprime_ws_norm = np.sqrt(vprime_ws_norm)

        return (vocab,vprime_ws,vprime_ws_norm)


    def score_new(self,s,t):
        '''
        Returns the similarity score
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


    def score(self,s,t):
        '''
        Returns the similarity score
        '''
        similar = namedtuple('Similar',['r1','r2','sim'])
        similarity=[]
        tfidfdict = self.builddict()
        for i,ti in enumerate(s.split(" ")):
            for j,tj in enumerate(t.split(" ")):
                dist = jf.jaro_winkler(ti,tj)
                if dist >= self.THRESHOLD:
                    similarity.append(similar(i,j,dist*tfidfdict.get(ti)*tfidfdict.get(tj)))

        similarity.sort(reverse=True,key=lambda x:x.sim)

        sused = np.array([False]*len(s),dtype=bool)
        tused = np.array([False]*len(t),dtype=bool)

        #check that the term are counted only once
        sim = 0.0
        for s in similarity:
            if(sused[s.r1] | tused[s.r2]):
                continue;
            sim+=s.sim
            sused[s.r1] = True
            tused[s.r2] = True
        return sim  


    def main():
        """ Driver program """
        s=Softtfidf()
        document1 = u'apoclapse now'
        document2 = u'apocalypse now'
        s.CORPUS.append(document1)
        s.CORPUS.append(document2)
        s.logger.info(s.score(document1,document2))



if __name__ == '__main__':
    main()

