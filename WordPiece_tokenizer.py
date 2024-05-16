import re
import html
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict
from tqdm import tqdm

class WordPieceTokenizer:
    """
    A tokenizer that implements the WordPiece tokenization algorithm. This tokenizer is designed to
    split text into meaningful subwords based on frequency statistics of subwords in a given corpus.
    The tokenizer also supports various preprocessing options to prepare text for tokenization.

    Attributes:
        vocab_stoi (Dict[str, int]): A dictionary mapping subwords to their indices.
        vocab_itos (List[str]): A list where the index represents the subword id and the value is the subword.
        meta_tokens (Dict[str, str]): Special tokens used for various metadata purposes (e.g., unknown, padding).
        preprocess_tokens (Dict[str, str]): Tokens used for indicating certain preprocessing states (e.g., uppercase).
        preprocess_args (Dict[str, bool]): Arguments dictating how text should be preprocessed.
        learn_wp_args (Dict[str, Any]): Configuration for learning the vocabulary, such as size and frequency requirements.
    """
    
    def __init__(self):
        """
        Initializes the WordPieceTokenizer with default settings for token handling and preprocessing.
        """
        
        self.vocab_stoi = None
        self.vocab_itos = None

        # Predefining tokens
        self.meta_tokens = dict(unk="<unk>", pad="<pad>", bos="<bos>", eos="<eos>")
        self.preprocess_tokens = dict(maj="<maj>", upp="<upp>")

        self.preprocess_args = dict(
            lowercase=True,
            spec_add_spaces=True,
            remove_useless_spaces=True,
            fix_html=True,
        )

        self.learn_wp_args = dict(
            vocab_size=30000,
            min_frequency=2,
            #special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
            required_chars=[],
            required_subwords=[],
            required_words=[],
            num_chars=-1,
            num_words=0
        )

    def _preprocess(self, text: str) -> str:
        """
        Applies preprocessing steps to text according to the tokenizer's configuration.

        Parameters:
            text (str): The text to preprocess.

        Returns:
            str: The preprocessed text.
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

    def _count_subwords(self, corpus: str) -> Counter:
        """
        Counts all possible substrings (subwords) in the corpus, used to build the vocabulary based on their frequencies.

        Parameters:
            corpus (str): The text corpus from which to count subwords.

        Returns:
            Counter: A counter object mapping each subword to its frequency in the corpus.
        """
        
        words = corpus.split()
        subword_counts = Counter()
        for word in words:
            for i in range(len(word)):
                for j in range(i + 1, len(word) + 1):
                    subword = word[i:j]
                    subword_counts[subword] += 1
        return subword_counts

    def _init_vocab(self, corpus: str, subword_counts: Counter, args: Dict) -> Tuple[Dict[str, int], List[str]]:
        """
        Initializes the vocabulary from the corpus based on the frequency of subwords and additional configuration.

        Parameters:
            corpus (str): The corpus text.
            subword_counts (Counter): A counter of subword frequencies.
            args (Dict): Additional arguments specifying requirements for the vocabulary.

        Returns:
            Tuple[Dict[str, int], List[str]]: A tuple containing the string-to-index mapping and index-to-string list.
        """
        
        tmp_vocab_itos = [v for v in self.meta_tokens.values()] + [v for v in self.preprocess_tokens.values()] + \
                         args['required_chars'] + args['required_subwords'] + args['required_words']

        if args['num_chars']:
            char_counts = Counter(corpus)
            if args['num_chars'] == -1:
                args['num_chars'] = len(char_counts)
            tmp_vocab_itos += sorted(char_counts, key=char_counts.get, reverse=True)[:args['num_chars']]
        if args['num_words']:
            if args['num_words'] == -1:
                args['num_words'] = len(subword_counts)
            tmp_vocab_itos += sorted(subword_counts, key=subword_counts.get, reverse=True)[:args['num_words']]

        tmp_lookup = set()
        vocab_itos = [x for x in tmp_vocab_itos if x not in tmp_lookup and tmp_lookup.add(x) is None]
        vocab_stoi = {s:i for i,s in enumerate(vocab_itos)}
        unk_i = vocab_stoi[self.meta_tokens['unk']]
        vocab_stoi = defaultdict(lambda: unk_i, vocab_stoi)

        return vocab_stoi, vocab_itos

    def _learn_wordpiece_vocab(self, corpus: str):
        """
        Learns the WordPiece vocabulary from a given corpus by applying preprocessing and counting subwords.

        Parameters:
            corpus (str): The text corpus from which to learn the vocabulary.
        """
        
        args = self.learn_wp_args

        corpus = self._preprocess(corpus)
        subword_counts = self._count_subwords(corpus)
        self.vocab_stoi, self.vocab_itos = self._init_vocab(corpus, subword_counts, args)

        num_subwords = args['vocab_size'] - len(self.vocab_itos)
        for _ in tqdm(range(num_subwords), desc="Learning WordPiece vocab"):
            most_frequent_subword = None
            highest_frequency = 0
            for subword, freq in subword_counts.items():
                if freq > highest_frequency and subword not in self.vocab_stoi:
                    highest_frequency = freq
                    most_frequent_subword = subword

            if most_frequent_subword is None:
                break

            self.vocab_stoi[most_frequent_subword] = len(self.vocab_itos)
            self.vocab_itos.append(most_frequent_subword)
            subword_counts[most_frequent_subword] = 0  # Reseting frequency to avoid re-adding

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes a given text into subwords according to the learned vocabulary.

        Parameters:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of token strings, where each token is either a recognized subword or a special token indicating an unknown subword.
        """
        
        tokens = []
        for word in text.split():
            start = 0
            sub_tokens = []
            while start < len(word):
                end = len(word)
                while start < end:
                    subword = word[start:end]
                    if subword in self.vocab_stoi:
                        if start > 0:
                            sub_tokens.append("##" + subword)
                        else:
                            sub_tokens.append(subword)
                        break
                    end -= 1
                if start == end:
                    sub_tokens.append(self.meta_tokens['unk'])
                    break
                start = end

            tokens.extend(sub_tokens)

        return tokens

    def detokenize(self, tokens: List[str]) -> str:
        """
        Converts a list of tokens back into a coherent string, effectively reversing the tokenization process.

        Parameters:
            tokens (List[str]): The list of tokens to convert back to text.

        Returns:
            str: The detokenized text, reconstructed from the tokens.
        """
        
        words = []
        current_word = ""
        for token in tokens:
            if token.startswith("##"):
                current_word += token[2:]
            else:
                if current_word:
                    words.append(current_word)
                current_word = token
        if current_word:
            words.append(current_word)
        return " ".join(words)

    def encode(self, text: str) -> List[int]:
        """
        Encodes a given text into a list of integers based on the vocabulary indices.

        Parameters:
            text (str): The text to encode.

        Returns:
            List[int]: A list of integer indices representing the tokens in the text.
        """
        
        tokens = self.tokenize(text)
        return [self.vocab_stoi[s if not s.startswith("##") else s[2:]] for s in tokens]

    def decode(self, encodings: List[int]) -> str:
        """
        Decodes a list of integer indices back into the original text, based on the vocabulary.

        Parameters:
            encodings (List[int]): The list of token indices to decode.

        Returns:
            str: The decoded text, reconstructed from the indices.
        """
        
        tokens = [self.vocab_itos[i] for i in encodings]
        return self.detokenize(tokens)

    def encode_text(self, text: str) -> Tuple[List[int], List[str]]:
        """
        Encodes text and provides both the indices and the tokenized form.

        Parameters:
            text (str): The text to encode.

        Returns:
            Tuple[List[int], List[str]]: A tuple containing a list of indices and a list of token strings.
        """
        
        tokens = self.tokenize(text)
        token_ids = [self.vocab_stoi[s if not s.startswith("##") else s[2:]] for s in tokens]
        return token_ids, tokens

    def save(self, path: Path):
        """
        Saves the tokenizer state to a file, allowing it to be reloaded later.

        Parameters:
            path (Path): The file path where the tokenizer state should be saved.
        """
        
        with path.open('wb') as f:
            pickle.dump((self.vocab_itos, dict(self.vocab_stoi)), f)  # Converting defaultdict to dict for serialization

    @classmethod
    def load(cls, path: Path):
        """
        Loads a tokenizer state from a file, restoring a previously saved tokenizer.

        Parameters:
            path (Path): The file path from which to load the tokenizer.

        Returns:
            WordPieceTokenizer: An instance of WordPieceTokenizer with the loaded state.
        """
        
        wpt = cls()
        with path.open('rb') as f:
            itos, stoi = pickle.load(f)
        wpt.vocab_itos = itos
        wpt.vocab_stoi = defaultdict(lambda: wpt.meta_tokens['unk'], stoi)
        return wpt