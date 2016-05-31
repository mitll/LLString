# mitll-string-match

This project contains the source for **llstring**, a soft string matching toolbox.  **llstring** provides:
* Basic Soft string matching via Levenshtein, Jaro-Winkler and Soft TF-IDF similarity scoring algorithms
* Roll-Your-Own soft string matching functionality for new data domains:
    * Full cross-validation suite (including Platt Scaling) for building classifiers based on raw string matching scores
    * TF-IDF dictionary training
    * Example models trained on English language social media data
* Text Normalization Tools (i.e. UTF-8 encoding normalization, tweet-speak normalization, etc.)

### Tweet Normalization example

```
515918-mitll:mitll-string-match go22670$ export PYTHONIOENCODING=utf-8
515918-mitll:mitll-string-match go22670$ python norm.py twitterSample.tsv  > twitterSampleNormed.tsv
```

which strips out @mentions, hash tags, urls, etc.

e.g. original:

```
==> twitterSample.tsv <==
pt	pensei que tinha perdido boina e caderno, mas na vdd, a gabriella que pegou, nojenta"
und	@ZEECorporate     A3) 5 continents #ZEE"
und	@eduarda_sam ui apxnd"
ka	თელავი ჩოგბურთის საერთაშორისო ტურნირს მასპინძლობს: თელავში ქალთა და მამაკაცთა სერთაშორისო ტურნირი ჩოგბურთში “T... http://t.co/vzu6LgoLeH"
km	អានេះទើបហៅថា មី ពិតប្រាកដព្រោះដុតមិនឆេះទេ ខុសពី មីយួន ដុតទៅឆេះរលួយដូច ជ័រកៅស៊ូ http://t.co/0HHJkFT1tD"
sv	@lisaginell Jaa verkligen,världens bästa typ som små hundar :')"
nl	Inpakken voor Rastede! :) Nu hoop ik eiglk dat alles in 1 weekendtas past..."
da	@koefoed Der fik han lige 500 XP."
fa	این بچه ها چه قدر تو وایبر گروه تشکیل میدن..یعنی همه جکان رو نمیتونن تو همون گروه کپی کنن؟:/"
it	C'è un umidità assurda💧💧💧"
pa	ਇਹ ਮੁਹੱਬਤ ਕਰਨ ਵਾਲੇ ਲੋਕ ਨਿਰੇ ਮੂਰਖ ਹੁੰਦੇ ਹਨ ਖ਼੍ਜ਼ਾਨੇ ਦੀ ਕੁੰਜੀ ਆਪੇ\nਹੀ ਕਿਸੇ ਬਿਹਾਨੇ ਨੂੰ ਸੌਂਪ ਦਿੰਦੇ ਹਨ,ਤੇ ਫ਼ੇਰ ਆਪ੍ਣੇ... http://t.co/PcfxbpfzxA"
und	heol"
und	Derbyshire 62/2 (21.4 ov) #cricket #fifa14 #FIFAWorldCup 267"
de	Ich habe 71,530 Goldmünzen gesammelt! http://t.co/PwPNZDFUVy #android, #androidgames, #gameinsight"
not_fi	@stevenlongo_33 oh myy 🙊"
```

to normalized:

```
==> twitterSampleNormed.tsv <==
pt	pensei que tinha perdido boina e caderno mas na vdd a gabriella que pegou nojenta
und	a3 5 continents
und	ui apxnd
ka	თელავი ჩოგბურთის საერთაშორისო ტურნირს მასპინძლობს თელავში ქალთა და მამაკაცთა სერთაშორისო ტურნირი ჩოგბურთში “t
km	អានេះទើបហៅថា មី ពិតប្រាកដព្រោះដុតមិនឆេះទេ ខុសពី មីយួន ដុតទៅឆេះរលួយដូច ជ័រកៅស៊ូ
sv	jaa verkligen världens bästa typ som små hundar :')
nl	inpakken voor rastede :) nu hoop ik eiglk dat alles in 1 weekendtas past
da	der fik han lige 500 xp
fa	این بچه ها چه قدر تو وایبر گروه تشکیل میدنیعنی همه جکان رو نمیتونن تو همون گروه کپی کنن؟:/
it	c'è un umidità assurda💧💧💧
pa	ਇਹ ਮੁਹੱਬਤ ਕਰਨ ਵਾਲੇ ਲੋਕ ਨਿਰੇ ਮੂਰਖ ਹੁੰਦੇ ਹਨ ਖ਼੍ਜ਼ਾਨੇ ਦੀ ਕੁੰਜੀ ਆਪੇ nਹੀ ਕਿਸੇ ਬਿਹਾਨੇ ਨੂੰ ਸੌਂਪ ਦਿੰਦੇ ਹਨ,ਤੇ ਫ਼ੇਰ ਆਪ੍ਣੇ
und	heol
und	derbyshire 62/2 214 ov 267
de	ich habe 71,530 goldmünzen gesammelt
not_fi	oh myy 🙊
```

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
