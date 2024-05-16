#Importing all the necessary packages
import re
import html
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
from tqdm import tqdm

class BpeTokenizer:
    """
    Byte Pair Encoding Tokenizer.

    This tokenizer utilizes Byte Pair Encoding (BPE) for tokenization.

    Attributes:
        vocab_stoi (dict): A dictionary mapping tokens to their corresponding integer indices.
        vocab_itos (dict): A dictionary mapping integer indices to their corresponding tokens.
        meta_tokens (dict): A dictionary containing predefined meta tokens and their replacements.
        preprocess_tokens (dict): A dictionary containing predefined preprocessing tokens and their replacements.
        preprocess_args (dict): A dictionary containing arguments for text preprocessing.
        learn_bpe_args (dict): A dictionary containing arguments for BPE learning.

    """
    def __init__(self):
        """
        Initialize BpeTokenizer with default parameters.

        Initializes vocab_stoi and vocab_itos as None.
        Predefines meta_tokens and preprocess_tokens.
        Sets default arguments for text preprocessing and BPE learning.
        """
        self.vocab_stoi = None
        self.vocab_itos = None

        # Predefine tokens
        self.meta_tokens = dict(unk="<unk>", pad="<pad>", bos="<bos>", eos="<eos>")
        self.preprocess_tokens = dict(maj="<maj>", upp="<upp>")

        self.preprocess_args = dict(
            lowercase=True,
            spec_add_spaces=True,
            remove_useless_spaces=True,
            fix_html=True,
        )

        self.learn_bpe_args = dict(
            vocab_size=30000,
            pairable_chars="ա-ֆԱ-Ֆ",  # Armenian character set
            unpairable_chars=None,
            unpairable_str="",

            required_chars=[],
            required_subwords=[],
            required_words=[],

            num_chars=-1,
            num_words=0,

            max_bpe_size=0,
            bpe_per_merge=10,
        )

    def _preprocess(self, text: str) -> str:
        """
            Preprocesses the corpus text for learning Byte Pair Encodings (BPEs).

            Args:
                text (str): The input corpus text to preprocess.

            Returns:
                str: The preprocessed text.

            Preprocessing steps:
                - Lowercase the text if specified in preprocess_args.
                - Add spaces around special characters if specified in preprocess_args.
                - Remove redundant spaces if specified in preprocess_args.
                - Fix HTML entities if specified in preprocess_args.
        """
        if self.preprocess_args['lowercase']:
            text = text.lower()

        if self.preprocess_args['spec_add_spaces']:
            text = re.sub(r'([/#\n])', r' \1 ', text)

        if self.preprocess_args['remove_useless_spaces']:
            text = re.sub(' {2,}', ' ', text)

        if self.preprocess_args['fix_html']:
            UNK = self.meta_tokens['unk']
            re1 = re.compile(r'  +')
            text = text.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
                'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
                '<br />', "\n").replace('\\"', '"').replace('<unk>', UNK).replace(' @.@ ', '.').replace(
                ' @-@ ', '-').replace(' @,@ ', ',').replace('\\', ' \\ ')
            return re1.sub(' ', html.unescape(text))

        return text

    def _count_words(self, corpus: str, args: Dict) -> Tuple[Counter, Counter]:
        """
            Count the occurrences of pairable and unpairable characters in the corpus.

            Args:
                corpus (str): The corpus text to analyze.
                args (Dict): A dictionary containing arguments for character counting.

            Returns:
                Tuple[Counter, Counter]: A tuple containing the counts of pairable characters and unpairable characters.
        """
        # Extracting arguments
        pairable_chars = args['pairable_chars']
        unpairable_chars = args['unpairable_chars']
        unpairable_str = args['unpairable_str']
        
        # Validating arguments
        if pairable_chars and unpairable_chars:
            raise ValueError("Only one of `pairable_chars` or `unpairable_chars` can be provided")
        
        # Counting occurrences of unpairable characters
        unpairable_counts = Counter(re.findall("|".join(unpairable_str), corpus))
        corpus = re.sub("|".join(unpairable_str),"", corpus)

        # Counting occurrences of pairable or unpairable characters based on provided arguments
        if pairable_chars:
            word_counts = Counter(re.findall(f"[{pairable_chars}]+", corpus))
            unpairable_counts.update(re.findall(f"[^{pairable_chars}]", corpus))
        elif unpairable_chars:
            word_counts = Counter(re.findall(f"[^{unpairable_chars}]+", corpus))
            unpairable_counts.update(re.findall(f"[{unpairable_chars}]", corpus))
        # If neither pairable_chars nor unpairable_chars are provided, default to Armenian character set
        else:
            word_counts = Counter(re.findall("[ա-ֆԱ-Ֆ]+", corpus))
            unpairable_counts.update(re.findall("^[^ա-ֆԱ-Ֆ]", corpus))

        return word_counts, unpairable_counts

    def _init_vocab(self, corpus: str, word_counts: Counter, args: Dict) -> Tuple[Dict[str, int], List[str]]:
        """
        Initialize the vocabulary from the corpus and word counts.

        Args:
            corpus (str): The corpus text.
            word_counts (Counter): A Counter containing word counts.
            args (Dict): A dictionary containing arguments for vocabulary initialization.

        Returns:
            Tuple[Dict[str, int], List[str]]: A tuple containing the vocabulary as a mapping from tokens to indices
            and a list of tokens.

        """
        # Initializing vocabulary with predefined tokens and required characters, subwords, and words
        tmp_vocab_itos = [v for v in self.meta_tokens.values()] + [v for v in self.preprocess_tokens.values()] + \
                     args['required_chars'] + args['required_subwords'] + args['required_words']
        
        # Including most frequent characters if specified by num_chars argument
        if args['num_chars']:
            char_counts = Counter(corpus)
            if args['num_chars'] == -1:
                args['num_chars'] = len(char_counts)
            tmp_vocab_itos += sorted(char_counts, key=char_counts.get, reverse=True)[:args['num_chars']]
            
        # Including most frequent words if specified by num_words argument
        if args['num_words']:
            if args['num_words'] == -1:
                args['num_words'] = len(word_counts)
            tmp_vocab_itos += sorted(word_counts, key=word_counts.get, reverse=True)[:args['num_words']]
            
        # Removing duplicates while maintaining order
        tmp_lookup = set()
        vocab_itos = [x for x in tmp_vocab_itos if x not in tmp_lookup and tmp_lookup.add(x) is None]
        
        # Creating mappings from tokens to indices
        vocab_stoi = {s:i for i,s in enumerate(vocab_itos)}
        
        # Setting index for unknown token
        unk_i = vocab_stoi[self.meta_tokens['unk']]
        vocab_stoi = defaultdict(lambda: unk_i, vocab_stoi)

        return vocab_stoi, vocab_itos

    def _learn_bpe_vocab(self, corpus: str):
        """
        Learn Byte Pair Encodings (BPE) vocabulary from the corpus.

        Args:
            corpus (str): The corpus text.

        """
        # Extracting BPE learning arguments
        args = self.learn_bpe_args

        # Preprocessing the corpus
        corpus = self._preprocess(corpus)
        
        # Counting words and unpairable characters
        word_counts, unpairable_counts = self._count_words(corpus, args)
        
        # Initializing vocabulary
        vocab_stoi, vocab_itos = self._init_vocab(corpus, word_counts, args)

        # Initializing word encodings
        word_encodings = {word: [c for c in word] for word in word_counts.keys()}
        
        # Determining number of BPE merges to perform
        num_bpe = args['vocab_size'] - len(vocab_itos)
        num_merges = num_bpe // args['bpe_per_merge']
        
        # Iterating over BPE merges
        for _ in tqdm(range(num_merges)):
            bp_counts = defaultdict(int)
            bp_words = defaultdict(set)
            
            # Iterating over words and their encodings
            for word, encodings in word_encodings.items():
                for bytepair in zip(encodings[:-1], encodings[1:]):
                    bp = "".join(bytepair)
                    # Checking if byte pair is not in vocabulary and meets max_bpe_size requirement
                    if bp not in vocab_stoi and (len(bp) <= args['max_bpe_size'] or not args['max_bpe_size']):
                        bp = " ".join(bytepair)
                        bp_counts[bp] += word_counts[word]
                        bp_words[bp].add(word)
                        
            # Breaking loop if no more byte pairs found
            if len(bp_counts) == 0:
                break
                
            # Selecting best byte pairs for merging
            best_bp = sorted(bp_counts, key=bp_counts.get, reverse=True)[:args['bpe_per_merge']]
            
            # Performing merges
            for bp in best_bp:
                merged_bp = bp.replace(" ", "")
                vocab_itos.append(merged_bp)
                vocab_stoi[merged_bp] = len(vocab_itos) - 1
                for word in bp_words[bp]:
                    word_encodings[word] = " ".join(word_encodings[word]).replace(bp, merged_bp).split(" ")

        # Converting defaultdict to dict for serialization
        self.vocab_stoi = dict(vocab_stoi)  
        self.vocab_itos = vocab_itos

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize the input text using the learned vocabulary.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: A list of tokens.

        """
        tokens = []
        token = None

        # Flags for tracking special cases
        maj_flag = False # Indicates a capitalized token
        upp_flag = False # Indicates a token with uppercase letters

        # Iterating over characters in the text
        for c in text:
            if token is not None:
                new_token = token + c.lower()
                
                # Checking if the extended token is not in the vocabulary, or if it meets certain conditions
                if (new_token not in self.vocab_stoi) or \
                        (len(token) > 1 and c.isupper() and maj_flag) or \
                        (c.isupper() and not maj_flag and not upp_flag) or \
                        (c.islower() and upp_flag):
                    
                    # If the extended token is not valid, appending the current token to the list of tokens
                    if maj_flag:
                        tokens.append(self.preprocess_tokens['maj'])
                        maj_flag = False
                    elif upp_flag:
                        tokens.append(self.preprocess_tokens['upp'])
                        upp_flag = False
                    tokens.append(token)
                    token = None
                else:
                    # If the extended token is valid, updating the current token
                    token = new_token
                    if c.isupper() and maj_flag:
                        upp_flag = True
                        maj_flag = False

            # If the current character is not in the vocabulary, appending it as a token
            if c.lower() not in self.vocab_stoi:
                tokens.append(c)

            # If there's no current token and the character is uppercase, starting a new token
            elif token is None:
                if c.isupper():
                    maj_flag = True
                    token = c.lower()
                else:
                    token = c
                    
        # Appending any remaining token
        if token:
            tokens.append(token)

        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        """
        Detokenize a list of tokens into a single string.

        Args:
            tokens (List[str]): A list of tokens to detokenize.

        Returns:
            str: The detokenized text.
        """
        for i, token in enumerate(tokens):
            if token == self.preprocess_tokens['maj']:
                tokens[i + 1] = tokens[i + 1].title()
            if token == self.preprocess_tokens['upp']:
                tokens[i + 1] = tokens[i + 1].upper()
        tokens = [token for token in tokens if token not in self.preprocess_tokens.values()]
        text = "".join(tokens)
        return text

    def encode(self, text: str) -> List[int]:
        """Encode the input text into a list of token indices."""
        return [self.vocab_stoi[s] for s in self.tokenize(text)]

    def decode(self, encodings: List[int]) -> str:
        """Decode a list of token indices into a text string."""
        return self.detokenize([self.vocab_itos[i] for i in encodings])

    def encode_text(self, text: str) -> Tuple[List[int], List[str]]:
        """Encode the input text into token indices and tokens."""
        token_ids = self.encode(text)
        tokens = [self.vocab_itos[i] for i in token_ids]
        return token_ids, tokens

    def save(self, path: Path):
        """Save the tokenizer to a file."""
        with path.open('wb') as f:
            pickle.dump((self.vocab_itos, dict(self.vocab_stoi)), f)  # Converting defaultdict to dict for serialization

    @classmethod
    def load(cls, path: Path):
        """
        Load the tokenizer from a file.

        Args:
            path (Path): The path to the file containing the tokenizer.

        Returns:
            BpeTokenizer: An instance of the BpeTokenizer loaded from the file.
            
        """
        # Initializing a new instance of BpeTokenizer
        bpet = cls()
        
        # Loading vocabulary from file
        with path.open('rb') as f:
            itos, stoi = pickle.load(f)
            
        # Setting vocabulary and handling unknown tokens
        bpet.vocab_itos = itos
        bpet.vocab_stoi = defaultdict(lambda: bpet.meta_tokens['unk'], stoi)
        return bpet
