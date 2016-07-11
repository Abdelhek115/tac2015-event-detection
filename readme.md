# Event Nugget Extraction using Deep Neural Networks
This repository contains the files for our Event Nugget Detection systems that was submitted to the TAC 2015 shared task on Event Nugget Detection.

We implemented a feed-forward network following the approach of Collobert et al., 'NLP (almost) from scratch' and trained it on the provided data.


In case you like the work, please cite the following paper:

```
@inproceedings{	reimers-gurevych:2015:TAC,
	author = {Nils Reimers and Iryna Gurevych},
	title = {Event Nugget Detection, Classification and Coreference Resolution using
Deep Neural Networks and Gradient Boosted Decision Trees},
	month = {November},
	year = {2015},
	booktitle = {Proceedings of the Eight Text Analysis Conference (TAC 2015) (to appear)},
	editor = {National Institute of Standards and Technology (NIST)},
	location = {Gaithersburg, Maryland, USA},
	research_area = {Ubiquitous Knowledge Processing},
	url = {https://www.ukp.tu-darmstadt.de/fileadmin/user_upload/Group_UKP/publikationen/2015/2015_TAC_Event_Nugget_Detection.pdf},
}
```

> **Abstract:** For the shared task of event nugget detection at TAC 2015 we trained a deep feed forward network achieving an official F1-score of 65.31% for plain annotations, 55.56% for event mention type and 49.16% for the realis value.
For the task of Event Coreference Resolution we prototyped a simple baseline using Gradient Boosted Decision Trees achieving an overall average CoNLL score of 70.02%.



Contact Person: [Nils Reimers](https://www.ukp.tu-darmstadt.de/people/doctoral-researchers/nils-reimers/?no_cache=1) 

http://www.ukp.tu-darmstadt.de/

http://www.tu-darmstadt.de/

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 


# Requirements 
* To run the code, you need Python 2.7 as well as Theano (tested on Theano 0.7).
* For the preprocessing, [Stanford CoreNLP](http://stanfordnlp.github.io/CoreNLP/index.html) is required. Download and unzip Stanford CoreNLP and store the jars in the `corenlp` folder.
* Levy's word embeddings are required (we used the word embeddings based on Dependency links). Download them from [Levys website](https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/) and unzip them in the `vocab` folder. Make sure `vocab/deps.words` exists. 
* For out of vocabulary tokens, the system uses a special 'UNKNOWN' token. For the word embeddings by Levy, you need to append this word embedding to your deps.words file by executing `cat padding_unknown_300d.txt >> deps.words` in the `vocab` folder.

# Executing the scripts
To train your own models, execute `TrainSENNA.py`. Given the config/config.txt file, this script trains a new models based on the train, development, and test files in the tacdata folder.

To execute the pre-trained model, run the script `RunModel.py`. This file reads in the input.txt files and adds event annotations using a BIO enconding. The output is stored in the output.txt file.

`RunModel.py` requires [Stanford CoreNLP](http://stanfordnlp.github.io/CoreNLP/index.html). The jars must be saved in the _corenlp_ folder.


# License 
This code is published under GPL version 3 or later. 
