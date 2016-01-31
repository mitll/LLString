#!/usr/bin/env python

# mitll_string_matcher.py
#
# Basic String Matching Techniques:
#   - Levenshtein Distance
#   - Jaro-Winkler 
#   - Soft TF-IDF
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

import jellyfish

from . import softtfidf
from .utilities.normalization import text_normalization as tt


class MITLLStringMatcher:
    """
    MIT-LL String Matcher Class:

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


    def __init__(self,stf_thresh=0.6):
        """
        Constructor
        """
        self.normalizer = tt.MITLLTextNormalizer()
        self.stf_thresh = stf_thresh


    def clean_string(self,s):
        """
        Strip leading characters, lower 
        """

        if isinstance(s,unicode):
            ss = s.lower()
        else:
            ss = unicode(s.lower(),"utf-8")

        if len(ss) == 0:
            ss = u''

        return ss


    def levenshtein_similarity(self,s,t):
        """
        Levenshtein Similarity 
        """
        
        ss = self.clean_string(s)
        tt = self.clean_string(t)

        Ns = len(ss); Nt = len(tt);

        lev_sim = 1.0 - (jellyfish.levenshtein_distance(ss,tt))/float(max(Ns,Nt))

        return lev_sim


    def jaro_winkler_similarity(self,s,t):
        """
        Jaro-Winkler Similarity
        """
        
        ss = self.clean_string(s)
        tt = self.clean_string(t)

        jw_sim = jellyfish.jaro_winkler(ss,tt)

        return jw_sim


    def soft_tfidf_similarity(self,s,t):
        """
        Soft TFIDF Similarity:

        This similarity measure is only meaningful when you have multi-word strings. 
        For single words, this measure will return 0.0
        """
        
        ss = self.clean_string(s)
        tt = self.clean_string(t)
        
        stf = softtfidf.Softtfidf(self.stf_thresh)
        #stf.set_threshold(self.stf_thresh)
        tfidf_sim1 = stf.score(ss,tt)

        stf2 = softtfidf.Softtfidf(self.stf_thresh)
        #stf2.set_threshold(self.stf_thresh)
        tfidf_sim2 = stf2.score(tt,ss)

        tfidf_sim = 0.5*(tfidf_sim1+tfidf_sim2)

        return tfidf_sim


    def main(self):
        """
        Main Function
        """
        self.logger.info("Entity-Match Test:")

        s = "ALI SHAHEED MOHAMMED"
        t = "ALI SAJID MUHAMMAD"

        self.logger.info("Levenshtein: {0}".format(self.levenshtein_similarity(s,t)))
        self.logger.info("Jaro-Winkler: {0}".format(self.jaro_winkler_similarity(s,t)))
        self.logger.info("Soft-TFIDF: {0}".format(self.soft_tfidf_similarity(s,t)))
        


if __name__ == "__main__":

    MITLLStringMatcher().main()

