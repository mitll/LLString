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

    THRESHOLD = 0.5

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

