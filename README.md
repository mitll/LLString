# mitll-string-match

This project contains the source for basic string matching in support of XDATA. 

Basic String Matching Techniques:
    - Levenshtein Distance
    - Jaro-Winkler 
    - Soft TF-IDF

This repo is intended to hold source only. Any raw data (or derived data products) must be stored locally.

### Prerequisites and Installation

This package requires both the jellyfish and sklearn pacakges which are both available via pip via: 

```shell
sudo pip install jellyfish
sudo pip install sklearn
```

### Running

The code to compute string match scores for an example string pair (i.e. `ALI SHAHEED MOHAMMED` and `ALI SAJID MUHAMMAD`) for each of the supported techniques is contained in the `main()` function of `mitll_string_match.py`. 

```shell
./mitll_string_match.py
Fri, 28 Aug 2015 10:05:55 INFO     Entity-Match Test:
Fri, 28 Aug 2015 10:05:55 INFO     0.72619047619
Fri, 28 Aug 2015 10:05:55 INFO     0.842063492063
Fri, 28 Aug 2015 10:05:55 INFO     0.201993092498
```

### License

Copyright 2015 MIT Lincoln Laboratory, Massachusetts Institute of Technology 

Licensed under the Apache License, Version 2.0 (the "License"); you may not use these files except in compliance with the License.

You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
