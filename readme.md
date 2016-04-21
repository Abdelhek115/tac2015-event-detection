This repository contains the files for our Event Nugget Detection systems that was submitted to the TAC 2015 shared task on Event Nugget Detection.

We implemented a feed-forward network following the approach of Collobert et al., 'NLP (almost) from scratch' and trained it on the provided data.

# Requirements 
* To run the code, you need Python 2.7 as well as Theano (tested on Theano 0.7).
* For the preprocessing, [http://stanfordnlp.github.io/CoreNLP/index.html](Stanford CoreNLP) is required. Download and unzip Stanford CoreNLP and store the jars in the `corenlp` folder.
* Levy's word embeddings are required (we used the word embeddings based on Dependency links). Download them from [https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/](Levys) website and unzip them in the `vocab` folder. Make sure `vocab/levy.vocab` exists.

# Executing the scripts
To train your own models, execute `TrainSENNA.py`. Given the config/config.txt file, this script trains a new models based on the train, development, and test files in the tacdata folder.

To execute the pre-trained model, run the script `RunModel.py`. This file reads in the input.txt files and adds event annotations using a BIO enconding. The output is stored in the output.txt file.

`RunModel.py` requires [http://stanfordnlp.github.io/CoreNLP/index.html](Stanford CoreNLP). The jars must be saved in the _corenlp_ folder.

# License 
This code is published under GPL version 3 or later. In case you like the work, please cite the following paper:
[https://www.ukp.tu-darmstadt.de/publications/details/?no_cache=1&tx_bibtex_pi1[pub_id]=TUD-CS-2015-1325](Event Nugget Detection, Classification and Coreference Resolution using Deep Neural Networks and Gradient Boosted Decision Trees)
Nils Reimers and Iryna Gurevych In: National Institute of Standards and Technology (NIST): Proceedings of the Eight Text Analysis Conference (TAC 2015), November 2015. 


