#!/usr/bin/env python

# mitll_string_match.py
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
import softtfidf


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


    def __init__(self):
        """
        Constructor
        """


    def levenshtein_similarity(self,s,t,agg):
        """
        Levenshtein Similarity:

        If multi-word string, for each word in s1, we find best matching
        word in s2; we then use the value of "agg" (e.g. "avg" or "max")
        to get a final aggregate score for the two strings

        More often than not, "avg" is likely the most appropriate aggregator.
        """
        
        su_split = unicode(s.lower(),"utf-8").split(); tu_split = unicode(t.lower(),"utf-8").split();

        sims = list()

        for su in su_split:
            Ns = len(su)
            max_sim = 0.0
            
            for tu in tu_split:
                Nt = len(tu)
                norm_lv = 1.0 - (jellyfish.levenshtein_distance(su,tu))/float(max(Ns,Nt))
                if norm_lv > max_sim:
                    max_sim = norm_lv
            
            sims.append(max_sim)

        if agg == "max":
            lev_sim = float(max(sims))
        else: # if not max, always do avg
            lev_sim = np.mean(sims)

        return lev_sim


    def jaro_winkler_similarity(self,s,t,agg):
        """
        Jaro-Winkler Similarity:

        If multi-word string, for each word in s1, we find best matching
        word in s2; we then use the value of "agg" (e.g. "avg" or "max")
        to get a final aggregate score for the two strings

        More often than not, "avg" is likely the most appropriate aggregator.
        """
        
        su_split = unicode(s.lower(),"utf-8").split(); tu_split = unicode(t.lower(),"utf-8").split();

        sims = list()

        for su in su_split:
            max_sim = 0.0
            
            for tu in tu_split:
                jw = jellyfish.jaro_winkler(su,tu)
                if jw > max_sim:
                    max_sim = jw
            
            sims.append(max_sim)


        if agg == "max":
            jw_sim = float(max(sims))
        else: # if not max, always do avg
            jw_sim = np.mean(sims)

        return jw_sim


    def soft_tfidf_similarity(self,s,t):
        """
        Soft TFIDF Similarity:

        This similarity measure is only meaningful when you have multi-word strings. 
        For single words, this measure will return 0.0
        """

        s = unicode(s.lower(),"utf-8"); t = unicode(t.lower(),"utf-8")
        
        stf = softtfidf.Softtfidf()
        stf.CORPUS.append(s); stf.CORPUS.append(t)
        tfidf_sim = stf.score(s,t)

        return tfidf_sim


    def main(self):
        """
        Main Function
        """
        self.logger.info("Entity-Match Test:")

        s = "ALI SHAHEED MOHAMMED"
        t = "ALI SAJID MUHAMMAD"

        self.logger.info(self.levenshtein_similarity(s, t, "avg"))
        self.logger.info(self.jaro_winkler_similarity(s, t, "avg"))
        self.logger.info(self.soft_tfidf_similarity(s,t))
        


if __name__ == "__main__":

    MITLLStringMatcher().main()

