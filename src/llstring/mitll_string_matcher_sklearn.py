#!/usr/bin/env python

# mitll_string_matcher_sklearn.py
#
# MITLLStringMatcheSklearn: sklearn compatible version of MITLLStringMatcher
# 
# MITLLSTringMatcher:
#     Basic String Matching Techniques:
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
import os, logging
import numpy as np
import cPickle as pickle

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array,column_or_1d
from sklearn.utils.multiclass import check_classification_targets
from sklearn.linear_model import LogisticRegression as LR

from .softtfidf import Softtfidf
from .mitll_string_matcher import MITLLStringMatcher
from .utilities import normalization as normutils


class MITLLStringMatcherSklearn(MITLLStringMatcher,BaseEstimator,ClassifierMixin):
    """
    MIT-LL String Matcher as Sklearn Estimator:

     String Matching Techniques:
       - Levenshtein Distance
       - Jaro-Winkler 
       - Soft TF-IDF
    """

    def __init__(self,algorithm='jw', stf_thresh=0.6, idf_model=None, text_normalizer = None):
        """ Initialize dict containing hyperparameters """
        MITLLStringMatcher.__init__(self) #init base class members (i.e. normalizer)

        self.hyparams = dict()
        self.hyparams['match_fcn'] = None
        self.hyparams['algo'] = algorithm
        self.hyparams['txt_normer'] = text_normalizer

        if algorithm == 'lev': #levenshtein
            self.hyparams['match_fcn'] = self.levenshtein_similarity

        elif algorithm== 'jw': #jaro-winkler
            self.hyparams['match_fcn'] = self.jaro_winkler_similarity

        elif algorithm== 'stf': #softtfidf
            self.hyparams['match_fcn'] = self.soft_tfidf_similarity
            self.hyparams['stf_thresh'] = stf_thresh
            self.hyparams['idf_model'] = idf_model


    #
    # Basic String Matching Functions
    #
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
    def validate_hyparams(self):
        """ Basic hyperparameter input validation"""
        
        if self.hyparams['algo'] not in set(['lev','jw','stf']):
            raise ValueError("Value of algorithm has to be either 'lev','jw' or 'stf'")

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
            #self.logger.info(u"un-normalized pair:({0},{1})".format(pair[0],pair[1])) 
            s = self.normalizer.normalize(pair[0])
            t = self.normalizer.normalize(pair[1])
            #self.logger.info(u"normalized pair:({0},{1})".format(s,t)) 
            #self.logger.info("lengths: ({0},{1})".format(len(s),len(t)))
            #self.logger.info(u"============================")

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


    def get_raw_similarities_old(self, X):
        """ Convert input to raw similarities """

        similarities = list()

        for pair in X:
            #s = pair[0]; t = pair[1]
            s = self.normalizer.normalize(pair[0])
            t = self.normalizer.normalize(pair[1])

            if (len(s) > 0) and (len(t) > 0):
                sim = self.hyparams['match_fcn'](s,t)
                similarities.append(sim)

        sims_array = np.asarray(similarities).reshape(-1,1)
        
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

        self.hyparams['algo'] = model_in['algo']
        self.hyparams['txt_normer'] = model_in['txt_normer']
        self.lr_ = model_in['calibration']
        if model_in['algo'] == 'stf':
            self.hyparams['stf_thresh'] = model_in['stf_thresh']
            self.hyparams['idf_model'] = model_in['idf_model']
        self.init_algorithm() #validate hyparams (we assume object not fit when load_model called)

        return self


    #def _validate_targets(self, y):
    #    """ Validate class labels provided in y"""
    #    y_ = column_or_1d(y, warn=True)
    #    check_classification_targets(y)
    #    cls, y = np.unique(y_, return_inverse=True)
    #    if len(cls) < 2:
    #        raise ValueError(
    #            "The number of classes has to be at least 2; got %d"
    #            % len(cls))

    #    self.classes_ = cls

    #    #return np.asarray(y, dtype=np.float64, order='C')
    #    return y
    #def set_X_feat_index(self,X_feat_index):
    #    """ Set X_feat_index """
    #    self.X_feat_index = X_feat_index


    #def get_raw_similarities_matrix(self, X):
    #    """ Convert input to raw similarities """

    #    similarities = list()

    #    for i in xrange(X.shape[0]):
    #        # Re-construct strings from sparse input X
    #        (junk,inds) = X[i][0].nonzero()

    #        s = self.X_feat_index[inds[0]];s = s.split("=")[1]
    #        t = self.X_feat_index[inds[1]];t = t.split("=")[1]

    #        if self.algorithm == 'lev': #'lev':levenshtein
    #            sim = self.levenshtein_similarity(s,t)
    #            similarities.append(sim)

    #        elif self.algorithm == 'jw': #'jw':jaro-winkler
    #            sim = self.jaro_winkler_similarity(s,t)
    #            similarities.append(sim)

    #        elif self.algorithm == 'stf': #'stf':softtfidf
    #            sim = self.soft_tfidf_similarity(s,t)
    #            similarities.append(sim)

    #        else:
    #            raise ValueError("Algorithm has to be either 'lev','jw' or 'stf'")

    #    s = np.asarray(similarities).reshape(-1,1)
    #    
    #    return s




    #
    # Learning
    #
    def fit(self,X,y):
        """ Fit string matching models to training data
        Assuming X is list of tuples: (('s1',t1'),...,('sN',tN'))
        """
        y = y[:] #shallow copy y, b/c in-place operations to follow

        # Initialize algorithm, validate parameters
        self.init_algorithm()

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
        """ Score (may not need this) """
        #s = self.get_raw_similarities(X)

        #return self.lr_.score(s,y,sample_weight)
        from sklearn.metrics import f1_score
        return f1_score(y,self.predict(X),sample_weight=sample_weight)


    #def transform(self):
    #    """ Score (may not need this) """

    # Inherited from MITLLStringMatcher
    #def clean_string(self,s):
    #def levenshtein_similarity(self,s,t):
    #def jaro_winkler_similarity(self,s,t):
    #def soft_tfidf_similarity(self,s,t):
    #def main(self):
    #    """
    #    Main Function
    #    """
    #    self.logger.info("Entity-Match Test:")

    #    s = "ALI SHAHEED MOHAMMED"
    #    t = "ALI SAJID MUHAMMAD"

    #    self.logger.info("Levenshtein: {0}".format(self.levenshtein_similarity(s,t)))
    #    self.logger.info("Jaro-Winkler: {0}".format(self.jaro_winkler_similarity(s,t)))
    #    self.logger.info("Soft-TFIDF: {0}".format(self.soft_tfidf_similarity(s,t)))
        


if __name__ == "__main__":

    MITLLStringMatcherSklearn().main()

