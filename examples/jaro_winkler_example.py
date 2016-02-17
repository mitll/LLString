#! /usr/bin/env python

# jaro_winklker_example.py
#
# Example script to demonstrate Jaro-Winkler string-match classifier
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
import os
import logging
import cPickle as pickle

from sklearn import cross_validation 

from llstring.matching.mitll_string_matcher import MITLLStringMatcher

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

    # Jaro-Winkler Matcher (from training data)
    matcher = MITLLStringMatcher(algorithm='jw', text_normalizer = 'latin')  # Initialize JW matcher
    matcher.fit(X_train,y_train)                                             # Fit matcher to training data
    matcher.save_model(os.path.join(exampledir,"data/output/models/english_socialmedia_jw.model"))
                                                                             # ^Save learned model out

    posts = matcher.predict_proba(X_test)           # Posterior probabilities of match
    preds = matcher.predict(X_test)                 # Predicted labels for test data
    confs = matcher.decision_function(X_test)       # Confidence (as distance to hyperplane)
    score = matcher.score(X_test,y_test)            # Return classificaton performance  
    raw_sims = matcher.get_raw_similarities(X_test) # Return raw similarity scores (not probabilities)

    # Jaro-Winkler Matcher (from pre-trained model)
    matcher2 = MITLLStringMatcher(algorithm='jw',text_normalizer = 'latin')
    matcher2.load_model(os.path.join(exampledir,"data/output/models/english_socialmedia_jw.model"))

    posts2 = matcher2.predict_proba(X_test)

    if (posts2 == posts2).all(): logger.info("Jaro-Winkler Test: Pass")
    else: logger.info("Jaro-Winkler Test: Fail")

    # Scoring an example string pair
    s = u"Abe Lincoln"; t = u"Abraham Lincoln Lab"
    post = matcher.predict_proba([(s,t)])[0][1]   # Posterior probability of match
    pred = matcher.predict([(s,t)])[0]            # Predicted label for pair
    logger.info("Example Match Posterior: {0}".format(post))

