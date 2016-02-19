#! /usr/bin/env python

# setup.py
#
# Setup and Install of llstring
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

# Imports 
from setuptools import setup, find_packages

# Setup
setup(name='llstring',
      version='0.0.1',
      description='MIT-LL String Processing and Matching Tools',
      url='https://g62code.llan.ll.mit.edu/cdagli/mitll-string-match',
      author='Charlie Dagli',
      author_email='dagli@ll.mit.edu',
      license='APLv2',
      packages=find_packages())

