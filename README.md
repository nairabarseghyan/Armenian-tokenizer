# Armenian Tokenizer
## Students: Anna Shaljyan, Naira Maria Barseghyan
- This project aims to develop a tokenization system for the Armenian language and train it using a corpus derived from Armenian Wikipedia. Tokenization is a crucial step in natural language processing (NLP) that involves breaking down text into individual units or tokens, which could be words, characters, or other meaningful elements.  The project will explore various tokenization methods that are suitable for the nuances of the Armenian language. By evaluating the effectiveness of these methods, the project aims to determine which approach best captures the linguistic characteristics and structures of Armenian text, leading to improved tokenization accuracy and performance. The ultimate goal is to create a robust tool that can accurately segment Armenian text into tokens, facilitating further NLP tasks such as parsing, part-of-speech tagging, and sentiment analysis.

  
### Prerequisites
- Python 3.11

### Setting up a Python Virtual Environment
To run this project, you will need Python 3.11. It is recommended to use a virtual environment to avoid conflicts with other packages. Follow these steps to set up your environment:

1. **Install Python 3.11**  
   Ensure that Python 3.11 is installed on your system. You can download it from [python.org](https://www.python.org/downloads/release/python-3110/).

2. **Create a Virtual Environment**  
   Open a terminal and run the following command:
   ```bash
   python3.11 -m venv venv
   ```
3. **Create a Virtual Environment**  
  * On Windows 
  ```bash
  venv/Scripts/Activate
  ```
  
  * On MacOS/Linux:
  ```bash
  source venv/bin/activate
  ```
  
4. **Install Required Packages**  
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure
```
Armenian-Tokenizer/
   │
   ├── reports/                                           # Report and assets
   │   ├── img/                                           # Wordcloud Images
   │   ├── csv/                                           # Vocabs saved in csv for easy inspection
   │   └── Armenian Tokenizer - Report.pdf                # Final Report
   │
   ├── models/                                            # Saved models of this project 
   │   ├── armenian_bpe_tokenizer.pkl                     # Custom BPE trained model
   │   ├── armenian_wordpiece_tokenizer.pkl               # Custom WordPiece trained model
   │   ├── armenian_sentencepiece_tokenizer.model         # SentencePiece trained model
   │   ├── armenian_sentencepiece_tokenizer.vocab         # SentencePiece trained vocab
   │   ├── bpe_tokenizer_1000rows.pkl                     # Tiktoken trained model
   │   ├── bpe_tokenizer_new.pkl                          # Tiktoken trained model
   │   └── bpe_tokenizer_trial.pkl                        # Tiktoken trained model
   │
   │
   ├── WordPiece_tokenizer.py                             # Custom WordPiece Tokenizer class 
   ├── BPE_tokenizer.py                                   # Custom BPE Tokenizer class
   ├── SentencePiece.ipynb                                # SentencePiece Training 
   ├── Training_tokenizers.ipynb                          # Training Custom Tokenizers
   ├── Testing_tokenizers.ipynb                           # Testing trained toeknizers (except tiktoken)
   ├── Tiktoken_training_1000rows.ipynb                   # Tiktoken Training     
   ├── Tiktoken-training-10rows.ipynb                     # Tiktoken Training     
   ├── Tiktoken_p50k_base.ipynb                           # Tiktoken Tokenization     
   ├── Tiktoken_cl100k_base.ipynb                         # Tiktoken Tokenization                  
   ├── README.md                                          # Read Me file
   └── requirements.txt                                   # Python dependencies for the project

```
    
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
- SentencePiece.ipynb: Contains training of SentencePiece BPE tokenier
- armenian_sentencepiece_tokenizer.models and .vocab: Contain SentencePiece model and vocab.
