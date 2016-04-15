# mitll-string-match

This project contains the source for **llstring**, a soft string matching toolbox.  **llstring** provides:
* Basic Soft string matching via Levenshtein, Jaro-Winkler and Soft TF-IDF similarity scoring algorithms
* Roll-Your-Own soft string matching functionality for new data domains:
    * Full cross-validation suite (including Platt Scaling) for building classifiers based on raw string matching scores
    * TF-IDF dictionary training
    * Example models trained on English language social media data
* Text Normalization Tools (i.e. UTF-8 encoding normalization, tweet-speak normalization, etc.)


### Prerequisites and Installation

This package is written in Python. For easiest installation, users are encouraged to use [Anaconda](https://www.continuum.io/why-anaconda), a lightweight package manager, environment manager and Python distribution. **llstring** is distributed with builds for multiple platforms and can be installed via:

```shell
conda install -c file://local/path/to/llstring/conda/build llstring
```

### Running

Example scripts highlighting **llstring** functionality can be found in the ```examples``` directory in the project root. This directory contains examples scripts and data for soft string matcher training, validation and testing on sample data. 


### License

Copyright 2015-2016 MIT Lincoln Laboratory, Massachusetts Institute of Technology 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License.

You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
