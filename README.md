# Armenian Tokenizer
## Students: Anna Shaljyan, Naira Maria Barseghyan
- This project aims to develop a tokenization system for the Armenian language and train it using a corpus derived from Armenian Wikipedia. Tokenization is a crucial step in natural language processing (NLP) that involves breaking down text into individual units or tokens, which could be words, characters, or other meaningful elements.  The project will explore various tokenization methods that are suitable for the nuances of the Armenian language. By evaluating the effectiveness of these methods, the project aims to determine which approach best captures the linguistic characteristics and structures of Armenian text, leading to improved tokenization accuracy and performance. The ultimate goal is to create a robust tool that can accurately segment Armenian text into tokens, facilitating further NLP tasks such as parsing, part-of-speech tagging, and sentiment analysis.

## Files that are in the repository:
- requirements.txt: Contains all the necessary packages for running used files
- BPE_tokenizer.py : Contains custom BPE class for Byte-Pair Encoding Tokenization
- Testing_tokenizers.ipynb: Contains saved models with examples in one ipynb (except tiktoken)
- Training_tokenizers.ipynb: Contains training of BPE and WordPiece
- Tiktoken-trainig-10rows.ipynb: Contains training on 10 articles of Armenian Wikipedia with vocab 400, saved under bpe_tokenizer_new.pkl
- Tiktoken_training_1000rows.ipynb: Contains training on 1000 articles of Armenian Wikipedia with vocab 400, saved under bpe_tokenizer_1000rows.pkl
- Tiktoken_cl100k_base.ipynb: Contains tokenization using tiktoken's cl100k_base model
- Tiktoken_p50k_base.ipynb: Contains tokenization using tiktoken's p50k_base model
- WordPiece_tokenizer.py: Contains custom WordPiece class for WordPiece Tokenization
- armenian_bpe_tokenizer.pkl: Contains custom BPE's trained model
- bpe_tokenizer_trial.pkl: Contains training on portion of Armenian Wikipedia with vocab 1000 using tiktoken
- inspecting_vocab.csv: Contains csv of saved vocabulary of custom BPE
- inspecting_vocab.xlsx: Contains excel of saved vocabulary of custom BPE
- wordcloud_1.png : wordcloud_5.png : Contain 5 wordclouds of resulted vocabulary of custom BPE
- Armenian Wikipedia Dataset.pdf: Contains pdf file mentioning two links to the main datasets (one is a newer version).
- Armenian Tokenizer - Report: Contains main report of the project
