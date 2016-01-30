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
import os
import logging
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_array,column_or_1d
from sklearn.utils.multiclass import check_classification_targets
from sklearn.linear_model import LogisticRegression as LR

import jellyfish

from .mitll_string_matcher import MITLLStringMatcher


class MITLLStringMatcherSklearn(MITLLStringMatcher,BaseEstimator,ClassifierMixin):
    """
    MIT-LL String Matcher as Sklearn Estimator:

     Basic String Matching Techniques:
       - Levenshtein Distance
       - Jaro-Winkler 
       - Soft TF-IDF
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


    def __init__(self,algorithm='jw',X_feat_index=None):
        """
        Constructor
        """
        self.algorithm = algorithm #'lev':levenshtein, 'jw':jaro-winkler or 'stf':softtfidf
        self.X_feat_index = X_feat_index #feature index to string mapping (at a corpus level)
        #NOTE: self.stf_thresh can be set in base class
        
    
    #
    # Utitlity Functions
    #
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

    
    def set_X_feat_index(self,X_feat_index):
        """ Set X_feat_index """
        self.X_feat_index = X_feat_index


    def get_raw_similarities_old(self, X):
        """ Convert input to raw similarities """

        similarities = list()

        for i in xrange(X.shape[0]):
            # Re-construct strings from sparse input X
            (junk,inds) = X[i][0].nonzero()

            s = self.X_feat_index[inds[0]];s = s.split("=")[1]
            t = self.X_feat_index[inds[1]];t = t.split("=")[1]

            if self.algorithm == 'lev': #'lev':levenshtein
                sim = self.levenshtein_similarity(s,t)
                similarities.append(sim)

            elif self.algorithm == 'jw': #'jw':jaro-winkler
                sim = self.jaro_winkler_similarity(s,t)
                similarities.append(sim)

            elif self.algorithm == 'stf': #'stf':softtfidf
                sim = self.soft_tfidf_similarity(s,t)
                similarities.append(sim)

            else:
                raise ValueError("Algorithm has to be either 'lev','jw' or 'stf'")

        s = np.asarray(similarities).reshape(-1,1)
        
        return s


    def get_raw_similarities(self, X):
        """ Convert input to raw similarities """

        similarities = list()

        for pair in X:

            s = pair[0]; t = pair[1];

            if self.algorithm == 'lev': #'lev':levenshtein
                sim = self.levenshtein_similarity(s,t)
                similarities.append(sim)

            elif self.algorithm == 'jw': #'jw':jaro-winkler
                sim = self.jaro_winkler_similarity(s,t)
                similarities.append(sim)

            elif self.algorithm == 'stf': #'stf':softtfidf
                sim = self.soft_tfidf_similarity(s,t)
                similarities.append(sim)

            else:
                raise ValueError("Algorithm has to be either 'lev','jw' or 'stf'")

        s = np.asarray(similarities).reshape(-1,1)
        
        return s

    #
    # Learning
    #
    def fit_old(self,X,y):
        """ Fit string matching models to training data """

        # Get string match scores
        s = self.get_raw_similarities(X)

        # Do Platt Scaling 
        self.lr = LR()
        self.lr.fit(s,y)

        return self


    def fit(self,X,y):
        """ Fit string matching models to training data
        Assuming X is list of tuples: (('s1',t1'),...,('sN',tN'))
        """

        # Get string match scores
        s = self.get_raw_similarities(X)

        # Do Platt Scaling 
        self.lr = LR()
        self.lr.fit(s,y)

        return self

    #
    # Inference
    # 
    def decision_functioni_old(self,X):
        """ Take input data, turn into decision """
        s = self.get_raw_similarities(X)

        return self.lr.decision_function(s)


    def predict(self,X):
        """ Class predictions """
        s = self.get_raw_similarities(X)

        return self.lr.predict(s)


    def predict_proba(self,X):
        """ Posterior match probabilities (need this for log-loss for CV """
        s = self.get_raw_similarities(X)

        return self.lr.predict_proba(s)

    #
    # Evaluate
    #
    def score(self,X,Y,sample_weight=None):
        """ Score (may not need this) """
        s = self.get_raw_similarities(X)

        return self.lr.score(s,y,sample_weight)


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

