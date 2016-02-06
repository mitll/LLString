#! /usr/bin/env python

# twitter_cv.py
#
# Create TP training data from boston twitter/instagram match data
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

#
# Imports 
#
import os,sys
import logging
import json
import cPickle as pickle

import gzip

#from utilities import text_normalization as tn
from llstring.utilities.normalization import text_normalization as tn
import llstring.mitll_string_matcher as llsm

#from ConfigParser import SafeConfigParser

#
# Logging
#
LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)

#
# Globals
#
# get projectdir following project dir conventions...
projectdir = os.path.realpath(__file__).split('src')[0]

# projectname via project dir conventions...
projectname = projectdir.split(os.sep)[-2:-1][0]

#current package name (i.e. containing directory name)
pkgname = os.path.realpath(__file__).split(os.sep)[-2]

#class name
classname = __name__

#
# Load Configurations (if necessary)
#
#parser = SafeConfigParser()
#
## Read configuration file for project
#parser.read(os.path.join(PROJECT_DIR,'src/config/{0}.ini'.format(PROJECT_NAME))
#
## Get project-level configurations
#project_var = self.parser.get(PROJECT_NAME,'sometarget')
#
## Get project-level configurations
#class_var = self.parser.get('twitter_cv','sometarget')


if __name__ == "__main__":

    #
    # Setup directories and files
    # 
    logger.info(projectdir)
    logger.info(projectname)
    logger.info(pkgname)
    logger.info(classname)

    fnamein_table = 'table_profiles.tsv.gz'
    fnamein_truth = 'truth_twitter_instagram_boston.txt'

    ## Input/Output directories
    twitter_table_indir = os.path.join(projectdir,'data/input/match_data_wmc/twitter/boston/tables')
    instagram_table_indir = os.path.join(projectdir,'data/input/match_data_wmc/instagram/boston/tables')
    truth_indir = os.path.join(projectdir,'data/input/match_data_wmc/truth')
    truth_outdir = os.path.join(projectdir,'data/input/cv_match_data')

    # Normalizer
    normer = tn.MITLLTextNormalizer()

    #
    # Truth 
    #
    twt2inst = dict()
    inst2twt = dict()
    fin = open(os.path.join(truth_indir,fnamein_truth),'r')
    for line in fin:
        line_split = line.split()
        tw_uname = normer.normalize_unicode_composed(unicode(line_split[0].strip(),'utf8'))
        inst_uname = normer.normalize_unicode_composed(unicode(line_split[1].strip(),'utf8'))
        twt2inst[tw_uname] = inst_uname
        inst2twt[inst_uname] = tw_uname
    fin.close()

    logger.info(twt2inst)

    #
    # Twitter Table
    #
    tw_uname2fullname = dict()
    fin = gzip.open(os.path.join(twitter_table_indir,fnamein_table),'rb')
    header = fin.readline()
    for line in fin:
        line_split = line.split('\t')
        tw_uname = normer.normalize_unicode_composed(unicode(line_split[8].strip(),'utf8'))
        tw_fullname = normer.normalize_unicode_composed(unicode(line_split[3].strip(),'utf8'))
        try:
            twt2inst[tw_uname]; #hacky, yet fast way to check for set membership
            tw_uname2fullname[tw_uname] = tw_fullname
        except:
            continue
        
    fin.close()

    logger.info(tw_uname2fullname)
    logger.info(len(tw_uname2fullname.keys()))

    #
    # Instagram Table
    #
    twt2inst_fullname = dict()
    fin = gzip.open(os.path.join(instagram_table_indir,fnamein_table),'rb')
    header = fin.readline()
    for line in fin:
        line_split = line.split('\t')
        inst_uname = normer.normalize_unicode_composed(unicode(line_split[8].strip(),'utf8'))
        inst_fullname = normer.normalize_unicode_composed(unicode(line_split[3].strip(),'utf8'))
        try:
            twt_uname = inst2twt[inst_uname]; #hacky, yet fast way to check for set membership
            twt2inst_fullname[tw_uname2fullname[twt_uname]] = inst_fullname
        except:
            continue
            
    fin.close()

    logger.info(twt2inst_fullname)
    logger.info(len(twt2inst_fullname))

    #
    # Grab all TP labeled pairs (TP and hard-negatives)
    #
    all_labeled_twt2inst_fullname = dict()
    fnameout = os.path.join(truth_outdir,"all_labeled_twt2inst_fullname.pckl")

    for s in twt2inst_fullname.keys():
        t = twt2inst_fullname[s]

        if (s == t) | (len(s) == 0) | (len(t) == 0):
            logger.info(u"(Ommiting: ({0},{1})".format(s,t))
            continue
        elif (s.lower() == t.lower()):
            logger.info(u"(Half-Counting: ({0},{1})".format(s,t))
            continue
        else:
            logger.info(u"\n({0},{1})".format(s,t))

            # record the TP pair
            all_labeled_twt2inst_fullname[s] = t


    pickle.dump(all_labeled_twt2inst_fullname,open(fnameout,"wb"))


    sys.exit()
    #
    # Filter some TP scores...
    #
    string_matcher = llsm.MITLLStringMatcher()
    
    labeled_twt2inst_fullname = dict()
    labeled_twt2inst_fullname_scores = dict()
    fnameout = os.path.join(truth_outdir,"labeled_twt2inst_fullname.pckl")

    for s in twt2inst_fullname.keys():
        t = twt2inst_fullname[s]

        lev = string_matcher.levenshtein_similarity(s,t)
        jw = string_matcher.jaro_winkler_similarity(s,t)
        soft_tfidf = string_matcher.soft_tfidf_similarity(s,t)

        if (s == t) | (len(s) == 0) | (len(t) == 0):
            logger.info(u"(Ommiting: {2:.2f},{3:.2f},{4:.2f})\t({0},{1})".format(s,t,lev,jw,soft_tfidf))
            keep = "n"
        elif (s.lower() == t.lower()):
            logger.info(u"(Half-Counting: {2:.2f},{3:.2f},{4:.2f})\t({0},{1})".format(s,t,lev,jw,soft_tfidf))
            keep = "y"
        else:
            logger.info(u"\n({2:.2f},{3:.2f},{4:.2f})\t({0},{1})".format(s,t,lev,jw,soft_tfidf))
            keep = raw_input("Keep Above?: ")

        if keep == "y":
            # record the TP pair
            labeled_twt2inst_fullname[s] = t

            # record the scores for that pair
            labeled_twt2inst_fullname_scores[s+u"_-_"+t] = dict()
            labeled_twt2inst_fullname_scores[s+u"_-_"+t]['lev'] = lev
            labeled_twt2inst_fullname_scores[s+u"_-_"+t]['jw'] = jw
            labeled_twt2inst_fullname_scores[s+u"_-_"+t]['stfidf'] = soft_tfidf
        
    pickle.dump([labeled_twt2inst_fullname,labeled_twt2inst_fullname_scores],open(fnameout,"wb"))



    ## make directories if they don't exist
    #if (not os.path.exists(corpus_outdir)): os.makedirs(corpus_outdir)
    #if (not os.path.exists(results_outdir)): os.makedirs(results_outdir)

    ##
    ## Setup data structures
    ##
    #feats_dict = dict()
    #uuids = list()

    ##
    ## Loop through something
    ##
    #for fname in os.listdir(feats_indir):
    #    fnamein = os.path.join(feats_indir,fname)


    ##
    ## Pickle out
    ##
    #corpus = dict()

    #corpus['feats_dict'] = feats_dict
    #corpus['uuids'] = uuids
    #corpus['feats'] = feats
    #corpus['tag2uuid'] = tag2uuid
    #corpus['uuid2tag'] = uuid2tag

    #fnameout_corpus = os.path.join(corpus_outdir,"corpus.pckl")
    #pickle.dump(corpus,open(fnameout_corpus,"wb"))

