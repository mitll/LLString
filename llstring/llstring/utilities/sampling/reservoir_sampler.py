#!/usr/bin/env python

# reservoir_sampler.py
#
# Perform uniform sampling from an (possibly infinite) input stream
# 
# Copyright 2015-2016 Massachusetts Institute of Technology, Lincoln Laboratory
# version 0.1
#
# author: Charlie K. Dagli
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

import random
import logging

class ReservoirSampler: 
    """ Class to perform uniform sampling from an input stream """

    # Logging
    LOG_LEVEL = logging.INFO
    logging.basicConfig(level=LOG_LEVEL,
                                format='%(asctime)s %(levelname)-8s %(message)s',
                                                    datefmt='%a, %d %b %Y %H:%M:%S')
    logger = logging.getLogger(__name__)
    

    def __init__(self,K):
        """ Constructor """
        self.K = K
        self.N = 0
        self.sample = list()


    def update_sample(self,item):
        """ Update sampler """
        self.N += 1
        
        if len(self.sample) < self.K:
            self.sample.append(item)
        else:
            s = int(random.random()*self.N)
            if s < self.K:
                self.sample[s] = item


    def get_sample(self):
        """ Return sample """
        return self.sample

