#! /usr/bin/env python

# llstring_training_example.py
#
# Example script to train string-match classifiers
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
from llstring.mitll_string_matcher import MITLLStringMatcher

from sklearn.feature_extraction import DictVectorizer
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

    DO_SAMPLE_TEST = False
    DO_IDF_TEST = False 
    DO_TRAINING_TEST = True 
    
    ###
    ### DATASET SAMPLING TESTS 
    ###

    #
    # Setup directories and files
    # 

    ## Input/Output directories
    twitter_table_indir = os.path.join(projectdir,'data/input/match_data_wmc/twitter/boston/tables')
    instagram_table_indir = os.path.join(projectdir,'data/input/match_data_wmc/instagram/boston/tables')
    outdir = os.path.join(projectdir,'data/output/models')

    # Fullname IDF sample filename
    fname_idf_train = os.path.join(outdir,"idf_training_data.txt")

    # Normalizer
    normer = latin_normalization.MITLLLatinNormalizer()

    # Samples
    sample_fullname = None


    if DO_SAMPLE_TEST:
        fnamein_table = 'table_profiles.tsv.gz'

        fnamein = os.path.join(twitter_table_indir,fnamein_table)
        N = 25000
        tw_fullname_sample = sample_from_tgz(fnamein,N)
        logger.info(tw_fullname_sample[0:100])
        logger.info("")

        fnamein = os.path.join(instagram_table_indir,fnamein_table)
        N = 25000
        inst_fullname_sample = sample_from_tgz(fnamein,N)
        logger.info(inst_fullname_sample[0:100])

        sample_fullname = tw_fullname_sample+inst_fullname_sample

        fw = open(fname_idf_train,"w")
        for fullname in sample_fullname:
            fw.write(fullname+"\n")
        fw.close()

    #
    # IDF training tests
    #
    if DO_IDF_TEST:
        # Assuming fname_idf_train file exists (i.e. we've already obtained a sample)
        if not sample_fullname:
            #load names from file
            sample_fullname = list()
            fo = open(fname_idf_train,"r")
            for line in fo:
                sample_fullname.append(line.rstrip())
            fo.close()

        #
        # Train to get IDF (from file handle)
        #
        idft = idf_trainer.IDFTrainer()
        idft.compute_idf(open(fname_idf_train,"r"))

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
        # Train to get IDF (from list)
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


    #
    # STRING MATCH CLASSIFIER TRAINING TESTS
    #
    if DO_TRAINING_TEST:

        # input training data 
        #y = list(); pairs = list()
        #y.append(1); pairs.append({'s':'stephen','t':'stevie'})
        #y.append(1); pairs.append({'s':'steve','t':'stevie'})
        #y.append(1); pairs.append({'s':'gary','t':'garrie'})
        #y.append(1); pairs.append({'s':'lloyd','t':'loyd'})
        #y.append(1); pairs.append({'s':'lawrence','t':'larry'})
        #y.append(1); pairs.append({'s':'michael','t':'michaeel'})
        #y.append(1); pairs.append({'s':'gloria','t':'loria'})
        #y.append(1); pairs.append({'s':'jeff','t':'jeof'})
        #y.append(1); pairs.append({'s':'calvin','t':'kalvin'})
        #y.append(0); pairs.append({'s':'steve','t':'comeau'})
        #y.append(0); pairs.append({'s':'ralph','t':'tony'})
        #y.append(0); pairs.append({'s':'susan','t':'andrea'})
        #y.append(0); pairs.append({'s':'michael','t':'elizabeth'})
        #y.append(0); pairs.append({'s':'marlon','t':'brando'})
        #y.append(0); pairs.append({'s':'walter','t':'morton'})
        #y.append(0); pairs.append({'s':'JD','t':'McPherson'})
        #y.append(0); pairs.append({'s':'goofball','t':'johnson'})
        #y.append(0); pairs.append({'s':'scott','t':'smith'})
        #y.append(0); pairs.append({'s':'andy','t':'loudermilk'})
        #y.append(0); pairs.append({'s':'renee','t':'smith'})

        #vec = DictVectorizer()
        #vec.fit_transform(pairs)
        #X = vec.fit_transform(pairs)
        #X_feat_index = vec.get_feature_names()

        #matcher = MITLLStringMatcher(algorithm='jw',X_feat_index=X_feat_index)
        #matcher.fit(X,np.asarray(y)); #make non-verbose


        # input training data as list
        #y = list(); training_list = list()
        #y.append(1); training_list.append(('stephen','stevie'))
        #y.append(1); training_list.append(('steve','stevie'))
        #y.append(1); training_list.append(('gary','garrie'))
        #y.append(1); training_list.append(('lloyd','loyd'))
        #y.append(1); training_list.append(('lawrence','larry'))
        #y.append(1); training_list.append(('michael','michaeel'))
        #y.append(1); training_list.append(('gloria','loria'))
        #y.append(1); training_list.append(('jeff','jeof'))
        #y.append(1); training_list.append(('calvin','kalvin'))
        #y.append(0); training_list.append(('steve','comeau'))
        #y.append(0); training_list.append(('ralph','tony'))
        #y.append(0); training_list.append(('susan','andrea'))
        #y.append(0); training_list.append(('michael','elizabeth'))
        #y.append(0); training_list.append(('marlon','brando'))
        #y.append(0); training_list.append(('walter','morton'))
        #y.append(0); training_list.append(('JD','McPherson'))
        #y.append(0); training_list.append(('goofball','johnson'))
        #y.append(0); training_list.append(('scott','smith'))
        #y.append(0); training_list.append(('andy','loudermilk'))
        #y.append(0); training_list.append(('renee','smith'))

        # load-in training data (tp matches and hard negatives)
        tp_training = pickle.load(open(os.path.join(projectdir,'data/input/cv_match_data/labeled_twt2inst_fullname.pckl'),'rb'))
        tp_keys = tp_training[0].keys()
        all_training = pickle.load(open(os.path.join(projectdir,'data/input/cv_match_data/all_labeled_twt2inst_fullname.pckl'),'rb'))
        y = list(); training_list = list();
        for s in all_training:
            training_list.append((s,all_training[s]))
            if s in tp_keys:
                y.append(1)
            else:
                y.append(0)

        #load-in more (easy) negatives to get pos/neg examples in balance
        N_negs = len(np.nonzero(y)[0]) - (len(y) - len(np.nonzero(y)[0]))
        count = 0
        fnamein = os.path.join(projectdir,"data/output/models/sampled_socialmedia_negative_pairs.txt")
        fo = open(fnamein,"r")
        for line in fo:
            if count < N_negs:
                line_split = unicode(line).rstrip().split("\t")
                training_list.append((line_split[0],line_split[1]));
                y.append(0)
                count += 1
            else:
                break
        fo.close()


        # Train/Test Splits
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(training_list,y,test_size=0.3,random_state=0)

        #Levenshtein
        matcher_lev = MITLLStringMatcher(algorithm='lev',text_normalizer = 'latin')
        matcher_lev.fit(X_train,y_train);
        matcher_lev.save_model(os.path.join(projectdir,"data/output/models/english_socialmedia_lev.model"))
        levout1 = matcher_lev.predict_proba(X_test)

        matcher_lev2 = MITLLStringMatcher(algorithm='lev',text_normalizer = 'latin')
        matcher_lev2.load_model(os.path.join(projectdir,"data/output/models/english_socialmedia_lev.model"))
        levout2 = matcher_lev2.predict_proba(X_test)

        if (levout1 == levout2).all(): logger.info("Levenshtein Test: Pass")
        else: logger.info("Levenshtein Test: Fail")

        #Jaro-Winkler
        matcher_jw = MITLLStringMatcher(algorithm='jw',text_normalizer = 'latin')
        matcher_jw.fit(X_train,y_train)
        matcher_jw.save_model(os.path.join(projectdir,"data/output/models/english_socialmedia_jw.model"))
        jwout1 = matcher_jw.predict_proba(X_test)

        matcher_jw2 = MITLLStringMatcher(algorithm='jw',text_normalizer = 'latin')
        matcher_jw2.load_model(os.path.join(projectdir,"data/output/models/english_socialmedia_jw.model"))
        jwout2 = matcher_jw2.predict_proba(X_test)

        if (jwout1 == jwout2).all(): logger.info("Jaro-Winkler Test: Pass")
        else: logger.info("Jaro-Winkler Test: Fail")

        #Soft TF-IDF
        idf_model = pickle.load(open(os.path.join(projectdir,"data/output/models/english_socialmedia_idf.model"),'rb'))
        matcher_stf = MITLLStringMatcher(algorithm='stf',idf_model=idf_model,text_normalizer = 'latin') #use default stf_thresh=0.6
        matcher_stf.fit(X_train,y_train)
        matcher_stf.save_model(os.path.join(projectdir,"data/output/models/english_socialmedia_stf.model"))
        stfout1 = matcher_stf.predict_proba(X_test)

        matcher_stf2 = MITLLStringMatcher(algorithm='stf',text_normalizer = 'latin')
        matcher_stf2.load_model(os.path.join(projectdir,"data/output/models/english_socialmedia_stf.model"))
        stfout2 = matcher_stf2.predict_proba(X_test)

        if (stfout1 == stfout2).all(): logger.info("Soft TF-IDF Test: Pass")
        else: logger.info("Soft TF-IDF Test: Fail")

        logger.info(matcher_lev.score(X_test,y_test))
        logger.info(matcher_jw.score(X_test,y_test))
        logger.info(matcher_stf.score(X_test,y_test))

        #
        # Grid-Search Example
        #
        matcher_stf = MITLLStringMatcher(algorithm='stf',idf_model=idf_model,text_normalizer = 'latin')
        param_grid = {'stf_thresh':[0.75,0.8,0.825]}
        clf = GridSearchCV(matcher_stf,param_grid,cv=5,verbose=2)
        clf.fit(X_train,y_train)
        for params, mean_score, scores in clf.grid_scores_:
            print mean_score

        logger.info(clf.score(X_test,y_test))


        #
        # Detailed Look
        # 
        logger.info("TRUE POSITIVE TRIALS")
        tp = np.nonzero(np.asarray(y_test) == 1); tp = tp[0]
        for ind in tp:
            s = X_test[ind][0]; t = X_test[ind][1]
            lev_score = matcher_lev.predict_proba([(s,t)])[0][1]
            jw_score = matcher_jw.predict_proba([(s,t)])[0][1]
            stf_score = matcher_stf.predict_proba([(s,t)])[0][1]
            logger.info(u"{0},{1},{2}\t({3},{4})".format(lev_score,jw_score,stf_score,s,t))

        logger.info("TRUE NEGATIVE TRIALS")
        tn = np.nonzero(np.asarray(y_test) == 0); tn = tn[0]
        for ind in tn:
            s = X_test[ind][0]; t = X_test[ind][1]
            lev_score = matcher_lev.predict_proba([(s,t)])[0][1]
            jw_score = matcher_jw.predict_proba([(s,t)])[0][1]
            stf_score = matcher_stf.predict_proba([(s,t)])[0][1]
            logger.info(u"{0},{1},{2}\t({3},{4})".format(lev_score,jw_score,stf_score,s,t))
        
        # django reinhart
        # tbone walker

        # OUTSTANDING:
        #[] move a bunch of this sample code to "examples" directory 

        # COMPLETE:
        #[x] change LR parameters to tack closer to zero and one; balanced targets
        #[x] put instance variable called "match_algo" (as dict) which tracks model parameters and which match function to use 
            #[x] test to see whether model loading works in soft-tf-idf matcher
            #[x] is soft-tf-idf matcher symmetric by default now? do some anectdotal testing...
        #[x] write out model parameters saveout method;
        #[x] also load model parameters method; 
        #[x] fix get_raw_similarities() to handle non-conforming pairs
        #[x] incorporate text normalizer hyperparameter into string matcher model?
        #[x] fix mutuable list problem in fit() 
        #[x] with normalizer as generic, fix problem where normalizer returns string of length 1, but it's punctuation that
        #   gets stop-listed when it goes to stf
        #[x] raise ValueError for normalized strings that are zero-length
        #[x] raise ValueError for strings that are all stop-list words 
        #[x] load-in the actual data (including negative pairs, which you have to write the code for)
            #[x] code to randomly sample negative pairs
        #[x] Just make one matcher class
        #[x] add unicode() before scoring (in get_raw_similarity())
        #[x] do the cross-validation to choose hyperparamters for STFIDF
        #   [x] fix init() so it works with grid search (re-factor self.hyparams)
        #[x] remove "sklearn" from matcher class name, etc.

        # MAYBE: 
        #[] sklearn class: function to load idf model from fname
        #[] also IDF train or load-in for stfidf
        #[] Train arabic name match models to release only to XDATA community
        #[] Lin's paired data? 

































