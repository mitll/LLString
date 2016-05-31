#! /usr/bin/env python

# levenshtein_example.py
#
# Example script to demonstrate Levenshtein string-match classifier
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
import sys
import io

from llstring.utilities.normalization.latin_normalization import *
import re

#
# Logging
#
LOG_LEVEL = logging.DEBUG
logging.basicConfig(level=LOG_LEVEL,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    normalizer = MITLLLatinNormalizer()

    skip = 1
    for file in sys.argv:
        if (skip == 1):
            skip = 0
        else: 
            with io.open(file,'r',encoding='utf8') as f:
                for line in f:
                    words = line.split()
                    first = words[0]
                    line = line[len(first):]
                    line = normalizer.normalize_unicode_composed(unicode(line))
                    line = normalizer.remove_html_markup(line)
                    line = re.sub(r'\#[a-zA-Z0-9_]+', ' ',line,flags=re.UNICODE) # remove hashtags
                    line = normalizer.remove_twitter_meta(line)
                    line = normalizer.remove_nonsentential_punctuation(line)
                    line = normalizer.remove_word_punctuation(line)
                    line = normalizer.remove_repeats(line)
                    line = normalizer.clean_string(line)
                    if (line == ' '): line = ''

                    both = words[0] + "\t" + line
                    print both
                    #print newline
        


