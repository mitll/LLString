#!/usr/bin/env python

# mitll_string_matcher.py
#
# MITLLSTringMatcher:
#     SKLEARN compatable classifier implementing string matching techniques:
#       - Levenshtein Distance
#       - Jaro-Winkler 
#       - Soft TF-IDF
# 
# Copyright 2015 Massachusetts Institute of Technology, Lincoln Laboratory
# version 0.1
#
# author: Charlie Dagli
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


# Imports
import logging
import numpy as np
import cPickle as pickle

import jellyfish

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score

from .softtfidf import Softtfidf
from ..utilities import normalization as normutils


class MITLLStringMatcher(BaseEstimator,ClassifierMixin):
    """
    MIT-LL String Matcher as Sklearn Estimator:

     String Matching Techniques:
       - Levenshtein Distance
       - Jaro-Winkler 
       - Soft TF-IDF
    """

    # Logging
    LOG_LEVEL = logging.INFO
    logging.basicConfig(level=LOG_LEVEL,
                                format='%(asctime)s %(levelname)-8s %(message)s',
                                                    datefmt='%a, %d %b %Y %H:%M:%S')
    logger = logging.getLogger(__name__)

    def __init__(self,algorithm='jw', stf_thresh=0.6, idf_model=None, text_normalizer = None):
        """ Initialize dict containing hyperparameters """

        self.algorithm = algorithm
        self.stf_thresh = stf_thresh
        self.idf_model = idf_model
        self.text_normalizer = text_normalizer


    #
    # Basic String Matching Functions
    #
    def levenshtein_similarity(self,s,t):
        """ Levenshtein Similarity """

        Ns = len(s); Nt = len(t);

        lev_sim = 1.0 - (jellyfish.levenshtein_distance(s,t))/float(max(Ns,Nt))

        return lev_sim


    def jaro_winkler_similarity(self,s,t):
        """ Jaro-Winkler Similarity """

        jw_sim = jellyfish.jaro_winkler(s,t)

        return jw_sim


    def soft_tfidf_similarity(self,s,t):
        """
        Soft TFIDF Similarity:

        This similarity measure is only meaningful when you have multi-word strings. 
        For single words, this measure will return 0.0
        """
        stf = self.hyparams['matcher'] #soft tf-idf object

        tfidf_sim = 0.5*(stf.score(s,t)+stf.score(t,s))

        return tfidf_sim


    #
    # Utitlity Functions
    #
    def init_hyparams(self):
        """ Initialize hyper-parameters dict """

        self.hyparams = dict()
        self.hyparams['match_fcn'] = None
        self.hyparams['algo'] = self.algorithm
        self.hyparams['txt_normer'] = self.text_normalizer

        if self.algorithm == 'lev': #levenshtein
            self.hyparams['match_fcn'] = self.levenshtein_similarity

        elif self.algorithm== 'jw': #jaro-winkler
            self.hyparams['match_fcn'] = self.jaro_winkler_similarity

        elif self.algorithm== 'stf': #softtfidf
            self.hyparams['match_fcn'] = self.soft_tfidf_similarity
            self.hyparams['stf_thresh'] = self.stf_thresh
            self.hyparams['idf_model'] = self.idf_model


    def validate_hyparams(self):
        """ Basic hyperparameter input validation"""
        
        if self.hyparams['algo'] not in set(['lev','jw','stf']):
            raise ValueError("Value of algorithm has to be either 'lev','jw' or 'stf'. Got {0}".format(self.hyparams['algo']))

        if self.hyparams['txt_normer'] not in set(['latin',None]):
            raise ValueError("The only value of txt_normer currently support is 'latin' (or None)")

        if self.hyparams['algo'] == 'stf':
            if (self.hyparams['stf_thresh'] < 0) | (self.hyparams['stf_thresh'] > 1):
                raise ValueError("Value of soft tf-idf's internal jaro-winkler threshold", \
                        "must be [0,1].")

            if self.hyparams['idf_model']:
                if set(self.hyparams['idf_model'].keys()) != set(['idf','corpus_vocab','oov_idf_val']):
                    raise ValueError("IDF model provided must contain only the following keys: ", \
                            "'idf', 'corpus_vocab', and 'oov_idf_val'.")

                if (not isinstance(self.hyparams['idf_model']['idf'],np.ndarray)) or \
                        (self.hyparams['idf_model']['idf'].dtype.type is not np.float64):
                    raise ValueError("idf_model['idf'] must be an np.ndarray of dtype np.float64")

                if not isinstance(self.hyparams['idf_model']['corpus_vocab'],dict):
                    raise ValueError("idf_model['corpus_vocab'] must be a dict.")

                if not isinstance(self.hyparams['idf_model']['oov_idf_val'],float):
                    raise ValueError("idf_model['oov_idf_val'] must be a float.")


    def init_algorithm(self):
        """ Validate hyperparameter inputs, init matcher object if neccessary"""
        
        self.validate_hyparams()

        # Initialize Soft TF-IDF matcher if needed
        if self.hyparams['algo'] == 'stf': #softtfidf
            self.hyparams['matcher'] = Softtfidf(self.hyparams['stf_thresh'],self.hyparams['idf_model'])

        if self.hyparams['txt_normer'] == 'latin':
            self.normalizer = normutils.latin_normalization.MITLLLatinNormalizer()
        else:
            self.normalizer = normutils.text_normalization.MITLLTextNormalizer() #generic normer

    
    def get_raw_similarities(self, X, y=None):
        """ Convert input to raw similarities """

        #make sure we have [0,1] class encoding in y
        if y:
            if set(y) != set((0,1)):
                raise ValueError("y expects class labels to be from {0,1}") 

        similarities = list()

        for i in xrange(len(X)):
            pair = X[i]
            s = unicode(self.normalizer.normalize(pair[0]),'utf-8')
            t = unicode(self.normalizer.normalize(pair[1]),'utf-8')

            if (len(s) > 0) and (len(t) > 0):
                sim = self.hyparams['match_fcn'](s,t)
                similarities.append(sim)
            else:
                similarities.append(0.0)
                if y: y[i] = -1 #set y-value of non-conforming pair to -1

        sims_array = np.asarray(similarities).reshape(-1,1)
        
        if y:
            return (sims_array,y)
        else:
            return sims_array


    def save_model(self,fnameout):
        """ Save model parameters out after fitting. """
        
        if self.lr_:
            model_out = dict()
            model_out['algo'] = self.hyparams['algo']
            model_out['txt_normer'] = self.hyparams['txt_normer']
            model_out['calibration'] = self.lr_
            if self.hyparams['algo'] == 'stf':
                model_out['stf_thresh'] = self.hyparams['stf_thresh']
                model_out['idf_model'] = self.hyparams['idf_model']

            pickle.dump(model_out,open(fnameout,"wb"))
            return self
        else:
            raise ValueError("save_model failed: No model has yet been fit or loaded.")


    def load_model(self,fnamein):
        """ Load model parameters. """
        model_in = pickle.load(open(fnamein,'rb')) # will throw I/O error if file not found

        self.init_hyparams() #initialize hyper-parameter dict

        self.hyparams['algo'] = model_in['algo']
        self.hyparams['txt_normer'] = model_in['txt_normer']
        self.lr_ = model_in['calibration']
        if model_in['algo'] == 'stf':
            self.hyparams['stf_thresh'] = model_in['stf_thresh']
            self.hyparams['idf_model'] = model_in['idf_model']

        self.init_algorithm() #validate hyparams (we assume object not fit when load_model called)

        return self


    #
    # Learning
    #
    def fit(self,X,y):
        """ Fit string matching models to training data
        Assuming X is list of tuples: (('s1',t1'),...,('sN',tN'))
        """
        y = y[:] #shallow copy y, b/c in-place operations to follow

        # Initialize hyper-parameter dict then algorithm
        self.init_hyparams(); self.init_algorithm()

        # Get string match scores
        (s,y) = self.get_raw_similarities(X,y)

        # Get rid of any non-conforming pairs
        data = zip(s,y)
        for pair in reversed(data): #iterate backwards to remove items from "data" 
                                    #so as not to mess up internal indexing of for-loop
            if pair[1] == -1: 
                data.remove(pair)

        (s,y) = zip(*data) 
        
        # Do Platt Scaling 
        self.lr_ = LR(penalty='l1',class_weight='balanced')
        self.lr_.fit(s,y)

        return self


    #
    # Inference
    # 
    def decision_function(self,X):
        """ Take input data, turn into decision """
        s = self.get_raw_similarities(X)

        return self.lr_.decision_function(s)


    def predict(self,X):
        """ Class predictions """
        s = self.get_raw_similarities(X)

        return self.lr_.predict(s)


    def predict_proba(self,X):
        """ Posterior match probabilities (need this for log-loss for CV """
        s = self.get_raw_similarities(X)

        return self.lr_.predict_proba(s)

    #
    # Evaluate
    #
    def score(self,X,y,sample_weight=None):
        """ Score matcher """
        return roc_auc_score(y,self.predict(X),sample_weight=sample_weight)

