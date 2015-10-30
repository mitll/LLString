#! /usr/bin/env python

# twitter_idf_train.py
#
# Example script to train dataset-specific IDF vector
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
import numpy as np
import os,sys
import logging
import json
import cPickle as pickle

import gzip

from llstring.utilities.normalization import latin_normalization
from llstring.utilities.sampling import reservoir_sampler
from llstring.training import idf_trainer

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

def sample_from_tgz(fnamein,N):
    """
    Scan-through data table (fixed schema), return uniform sample
    """

    # Setup Sampler
    sampler_fullname = reservoir_sampler.ReservoirSampler(N)

    # Sample from Table
    fin = gzip.open(fnamein,'rb')
    header = fin.readline()
    for line in fin:
        line_split = line.split('\t')
        
        fullname = normer.normalize_unicode_composed(unicode(line_split[3].strip(),'utf8'))
        if len(fullname) > 0:
            sampler_fullname.update_sample(fullname)

    fin.close()

    fullname_sample = sampler_fullname.get_sample()

    # Do Full Text Normalization on the sampled names
    for ind in xrange(len(fullname_sample)):
        fullname_sample[ind] = normer.normalize(fullname_sample[ind])

    return fullname_sample


if __name__ == "__main__":

    #
    # Setup directories and files
    # 
    fnamein_table = 'table_profiles.tsv.gz'

    ## Input/Output directories
    twitter_table_indir = os.path.join(projectdir,'data/input/match_data_wmc/twitter/boston/tables')
    instagram_table_indir = os.path.join(projectdir,'data/input/match_data_wmc/instagram/boston/tables')
    outdir = os.path.join(projectdir,'data/output/models')

    # Normalizer
    normer = latin_normalization.MITLLLatinNormalizer()

    fnamein = os.path.join(twitter_table_indir,fnamein_table)
    N = 25000
    tw_fullname_sample = sample_from_tgz(fnamein,N)
    logger.info(tw_fullname_sample[0:100])
    logger.info("")

    fnamein = os.path.join(instagram_table_indir,fnamein_table)
    N = 25000
    inst_fullname_sample = sample_from_tgz(fnamein,N)
    logger.info(inst_fullname_sample[0:100])

    #
    # Write sample out to file
    #
    fnameout = os.path.join(outdir,"idf_training_data.txt")
    sample_fullname = tw_fullname_sample+inst_fullname_sample

    fw = open(fnameout,"w")
    for fullname in sample_fullname:
        fw.write(fullname+"\n")
    fw.close()

    #
    # Train to get IDF
    #
    idft = idf_trainer.IDFTrainer()
    idft.compute_idf(open(fnameout,"r"))

    #
    # Sniff tests (note: anecdotally, high IDF corresponds to unames being used as fullnames)  
    #
    index2name = dict()
    for name in idft.CORPUS_VOCAB.keys():
        index2name[idft.CORPUS_VOCAB[name]] = name

    low_idf_inds = np.argsort(idft.LOG_IDF)
    for i in range(0,50):
        logger.info("{0},{1}".format(index2name[low_idf_inds[i]],idft.LOG_IDF[low_idf_inds[i]]))

    logger.info("")

    high_idf_inds = np.argsort(-idft.LOG_IDF)
    for i in range(0,50):
        logger.info("{0},{1}".format(index2name[high_idf_inds[i]],idft.LOG_IDF[high_idf_inds[i]]))

    logger.info("")

    #
    # Train to get IDF
    #
    sample_fullname2 = list()
    sample_fullname2 = sample_fullname
        
    idft = idf_trainer.IDFTrainer()
    idft.compute_idf(sample_fullname2)

    #
    # Sniff tests (note: anecdotally, high IDF corresponds to unames being used as fullnames)  
    #
    index2name = dict()
    for name in idft.CORPUS_VOCAB.keys():
        index2name[idft.CORPUS_VOCAB[name]] = name

    low_idf_inds = np.argsort(idft.LOG_IDF)
    for i in range(0,50):
        logger.info("{0},{1}".format(index2name[low_idf_inds[i]],idft.LOG_IDF[low_idf_inds[i]]))

    logger.info("")

    high_idf_inds = np.argsort(-idft.LOG_IDF)
    for i in range(0,50):
        logger.info("{0},{1}".format(index2name[high_idf_inds[i]],idft.LOG_IDF[high_idf_inds[i]]))

    logger.info(len(idft.LOG_IDF))
