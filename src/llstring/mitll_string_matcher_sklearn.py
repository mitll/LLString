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


    def __init__(self,algorithm='jw',stf_thresh=0.6):
        """
        Constructor
        """
        self.algorithm = algorithm #'lev':levenshtein, 'jw':jaro-winkler or 'stf':softtfidf
        self.jw_thresh = jw_thresh
        
    def fit(self,X,y):
        """
        Fit string matching models to training data
        """
        self.logger.info("nothing yet...")
        self.classes_, y = np.unique(y, return_inverse=True)



        return self


    def decision_function(self,X):
        """ Take input data, turn into decision """


    def predict(self,X):
        """ Class predictions """
        D = self.decision_function(X)
        return self.classes_[np.argmax(D, axis=1)]


    def predict_proba(self,X):
        """ Posterior match probabilities (need this for log-loss for CV """

    def score(self):
        """ Score (may not need this) """


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

