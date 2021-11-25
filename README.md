This repo contains some experiments to explore the outputs of Morphemepiece
**currently most paths are hard coded, please adapt that according to your needs**

# Scripts

## tokenize-text.R 

Simple wrapper to have a commandline interface to morphemepiece

## wiki-tokenize.py

Script which tokenizes Hugging Face's English Wikipedia dataset with the BertTokenizer and with MorphemePiece for further analysis

## summary-stats.py

Computes summary stats over the tokenized datsets to compare the tokenization results.

Stats include:

* Vocabulary size of both tokenizers on the dataset
* Procentual overlap of both tokenizers' vocabulary 
* Average document level stats comprising the mean overlap in tokens in both representations and the mean vocabulary size needed to represent documents
* The top n tokens of every tokenizer
* The top n distinct tokens only present in a single tokenizer
* Plot of the distribution of the top 500 tokens



