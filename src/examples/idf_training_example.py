#! /usr/bin/env python

# idf_training_example.py
#
# Example script to learn IDF from a training corpus
# 
# Copyright 2016 Massachusetts Institute of Technology, Lincoln Laboratory
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
import os, logging
import numpy as np

from llstring.training import idf_trainer

#
# Logging
#
LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


def idf_display(idft,N=10):
    """ Print tokens with highest and lowest IDF

        Anecdotally, high IDF corresponds to unames being used as fullnames

        Input:  idft:   trained IDFTrainer instance
                N:      Number of tokens to print      
    """

    index2name = dict()
    for name in idft.CORPUS_VOCAB.keys():
        index2name[idft.CORPUS_VOCAB[name]] = name

    logger.info("")
    logger.info("TOKENS CORRESPONDING TO LOWEST IDF VALUES")
    logger.info("=========================================")
    low_idf_inds = np.argsort(idft.LOG_IDF)
    for i in range(0,N):
        logger.info("{0},{1}".format(index2name[low_idf_inds[i]],idft.LOG_IDF[low_idf_inds[i]]))

    logger.info("")

    logger.info("TOKENS CORRESPONDING TO HIGHEST IDF VALUES")
    logger.info("==========================================")
    high_idf_inds = np.argsort(-idft.LOG_IDF)
    for i in range(0,N):
        logger.info("{0},{1}".format(index2name[high_idf_inds[i]],idft.LOG_IDF[high_idf_inds[i]]))

    logger.info("")


if __name__ == "__main__":
    
    # Input and Output Filenames
    exampledir = os.path.dirname(os.path.realpath(__file__))
    fnamein = os.path.join(exampledir,"data/input/idf_training_data.txt")
    fnameout = os.path.join(exampledir,"data/output/models/english_socialmedia_idf.pckl")

    #
    # Train IDF from file handle
    # (i.e. for large training sets)
    #
    idft = idf_trainer.IDFTrainer()
    idft.compute_idf(open(fnamein,"r"))
    idft.save_model(fnameout)

    idf_display(idft,20)

    #
    # Train IDF from python list instance
    # (i.e. for training sets that can fit in memory)
    #

    # load-in training data
    training_data = list()
    fo = open(fnamein,"r")
    for line in fo: training_data.append(line.rstrip())
    fo.close()
        
    # compute IDF
    idft = idf_trainer.IDFTrainer()
    idft.compute_idf(training_data)
    idft.save_model(fnameout)

    idf_display(idft,20)

