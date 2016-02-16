#! /usr/bin/env python

# soft_tfidf_example.py
#
# Example script to demonstrate Soft TF-IDF string-match classifier
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
import numpy as np
import os,sys
import logging
import json
import cPickle as pickle


import gzip

from llstring.utilities.normalization import latin_normalization
from llstring.utilities.sampling import reservoir_sampler
from llstring.training import idf_trainer
from llstring.matching.mitll_string_matcher import MITLLStringMatcher

from sklearn import cross_validation 
from sklearn.grid_search import GridSearchCV

#
# Logging
#
LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # Input and Output Filenames
    exampledir = os.path.dirname(os.path.realpath(__file__))
    fnamein = os.path.join(exampledir,"data/input/match_training_data.pckl")

    # Load Training Data
    train = pickle.load(open(fnamein,"rb"))
    X = train['X'] #string pairs 
    y = train['y'] #corresponding labels (1:match, 0:no-match)

    # Train/Test Splits (via sklearn)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)

    #
    # Soft TF-IDF Matcher (from training data)
    #
    idf_model = pickle.load(open(os.path.join(exampledir,"data/output/models/english_socialmedia_idf.pckl"),'rb'))
                                        # ^ see idf_training_example.py
    stf_thresh = 0.6                    # Internal JW threshold                                                                        
    matcher = MITLLStringMatcher(algorithm='stf',text_normalizer='latin',idf_model=idf_model,stf_thresh=stf_thresh)
                                        # ^ Initialize Soft TF-IDF matcher
    matcher.fit(X_train,y_train)        # Fit matcher to training data
    fname_model = os.path.join(exampledir,"data/output/models/english_socialmedia_stf_{0}.model".format(stf_thresh)) 
                                        # ^ Model-out filename
    matcher.save_model(fname_model)     # Save model out

    posts = matcher.predict_proba(X_test)           # Posterior probabilities of match
    preds = matcher.predict(X_test)                 # Predicted labels for test data
    confs = matcher.decision_function(X_test)       # Confidence (as distance to hyperplane)
    score = matcher.score(X_test,y_test)            # Return classificaton performance  
    raw_sims = matcher.get_raw_similarities(X_test) # Return raw similarity scores (not probabilities)

    # Scoring an example string pair
    s = u"Abe Lincoln"; t = u"Abraham Lincoln Lab"
    post = matcher.predict_proba([(s,t)])[0][1]   # Posterior probability of match
    pred = matcher.predict([(s,t)])[0]            # Predicted label for pair
    logger.info("Example Match Posterior: {0}".format(post))

    #
    # Soft TF-IDF Matcher (from pre-trained model)
    #
    matcher2 = MITLLStringMatcher(algorithm='stf',text_normalizer='latin',stf_thresh=0.6)
                                                                          # ^ Initialize Soft TF-IDF matcher
    matcher2.load_model(os.path.join(exampledir,"data/output/models/english_socialmedia_stf_{0}.model".format(stf_thresh)))
                                                                          # ^ Load-in model
    posts2 = matcher2.predict_proba(X_test)                               # Posterior probabilities of match

    if (posts2 == posts2).all(): logger.info("Soft TF-IDF Test: Pass")
    else: logger.info("Soft TF-IDF Test: Fail")

    #
    # Soft TF-IDF w/ hyper-parameter tuning (via sklearn Grid-Search) Example
    #
    matcher_stub = MITLLStringMatcher(algorithm='stf',idf_model=idf_model,text_normalizer = 'latin')
                                                                    # ^ Initialize Soft TF-IDF matcher stub
    param_grid = {'stf_thresh':[0.4,0.5,0.6,0.7,0.8,0.9]}           # Setup hyper-parameter grid
    cv_matcher = GridSearchCV(matcher_stub,param_grid,cv=5,verbose=2)   # Initialize GridSearchCV matcher of type
                                                                    #  MITLLStringMatcher(algorithm='stf')
    cv_matcher.fit(X_train,y_train)                                    # Re-train model on all training data using 
                                                                    #  (model fit to best performing hyper-parameter
                                                                    #   combination)

    fname_model = os.path.join(exampledir,"data/output/models/english_socialmedia_stf_optimal.model") 
                                                        # ^ Model-out filename
    cv_matcher.best_estimator_.save_model(fname_model)  # Save optimal model out

    logger.info("Best stf_thresh found by CV: {0}".format(cv_matcher.best_params_['stf_thresh']))

