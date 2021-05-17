# -*- coding: utf-8 -*-

"""Automated Tool for Optimized Modelling (ATOM).

Author: Mavs
Description: Module containing estimators for NLP.

"""

# Standard packages
import re
import nltk
import unicodedata
import pandas as pd
from string import punctuation
from typeguard import typechecked
from typing import Union, Optional
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    HashingVectorizer,
)
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

# Own modules
from .data_cleaning import TransformerMixin
from .basetransformer import BaseTransformer
from .utils import (
    SCALAR, SEQUENCE_TYPES, X_TYPES, Y_TYPES, check_is_fitted,
    get_corpus, composed, crash, method_to_log,
)


class TextCleaner(BaseEstimator, TransformerMixin, BaseTransformer):
    """Applies standard text cleaning to the corpus.

    Transformations include normalizing characters and dropping
    noise from the text (emails, HTML tags, URLs, etc...). The
    transformations are applied on the column named `Corpus`, in
    the same order the parameters are presented. If there is no
    column with that name, an exception is raised.

    Parameters
    ----------
    decode: bool, optional (default=True)
        Whether to decode unicode characters to their ascii
        representations.

    lower_case: bool, optional (default=True)
        Whether to convert all characters to lower case.

    drop_email: bool, optional (default=True)
        Whether to drop email addresses from the text.

    drop_url: bool, optional (default=True)
        Whether to drop URL links from the text.

    drop_html: bool, optional (default=True)
        Whether to drop HTML tags from the text. This option is
        particularly useful if the data was scraped from a website.

    drop_emojis: bool, optional (default=True)
        Whether to drop emojis from the text.

    drop_numbers: bool, optional (default=False)
        Whether to drop numbers from the text.

    drop_punctuation: bool, optional (default=True)
        Whether to drop punctuations from the text.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, Logger or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    Attributes
    ----------
    email: str
        Regex used to search for email addresses. The default
        value is `r"[\w.-]+@[\w-]+\.[\w.-]+"`.

    url: str
        Regex used to search for URLs The default value is
        `r"https?://\S+|www\.\S+"`.

    html: str
        Regex used to search for html tags. The default value
        is `r"<.*?>"`.

    emojis: str
        Regex used to search for emojis. The default value is
        `r":[a-z_]+:"`.

    numbers: str
        Regex used to search for numbers. The default value is
        `r"\b\d+\b"`. Note that numbers annexed to words are not
        removed.

    """

    @typechecked
    def __init__(
        self,
        decode: bool = True,
        lower_case: bool = True,
        drop_email: bool = True,
        drop_url: bool = True,
        drop_html: bool = True,
        drop_emojis: bool = True,
        drop_numbers: bool = True,
        drop_punctuation: bool = True,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.decode = decode
        self.lower_case = lower_case
        self.drop_email = drop_email
        self.drop_url = drop_url
        self.drop_html = drop_html
        self.drop_emojis = drop_emojis
        self.drop_numbers = drop_numbers
        self.drop_punctuation = drop_punctuation

        # Regular expressions to filter
        self.email = r"[\w.-]+@[\w-]+\.[\w.-]+"
        self.url = r"https?://\S+|www\.\S+"
        self.html = r"<.*?>"
        self.emojis = r":[a-z_]+:"
        self.numbers = r"\b\d+\b"
        self.punctuation = punctuation

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the transformations to the data.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features). If X is
            not a pd.DataFrame, it should be composed of a single
            feature containing the text documents. Each document is
            expected to be a string.

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        X: pd.DataFrame
            Transformed corpus.

        """
        def to_ascii(row):
            """Convert unicode string to ascii."""
            try:
                row.encode("ASCII", errors="strict")
            except UnicodeEncodeError:
                norm = unicodedata.normalize("NFKD", row)
                return "".join([c for c in norm if not unicodedata.combining(c)])
            else:
                return row  # If no unicode chars, return unchanged

        def drop_regex(regex):
            """Find and remove a regex expression from the text."""
            counts, docs = 0, 0
            for i, row in enumerate(X[corpus]):
                occurrences = regex.findall(row)
                if occurrences:
                    docs += 1
                    counts += len(occurrences)
                    for elem in occurrences:
                        X[corpus][i] = X[corpus][i].replace(elem, "", 1)

            return counts, docs

        X, y = self._prepare_input(X, y)
        corpus = get_corpus(X)

        self.log("Filtering the corpus...", 1)

        if self.decode:
            X[corpus] = X[corpus].apply(lambda row: to_ascii(row))
            self.log(" --> Decoding unicode characters to ascii.", 2)

        if self.lower_case:
            X[corpus] = X[corpus].str.lower()
            self.log(" --> Converting text to lower case.", 2)

        if self.drop_email:
            counts, docs = drop_regex(re.compile(self.email))
            self.log(f" --> Dropping {counts} emails from {docs} documents.", 2)

        if self.drop_url:
            counts, docs = drop_regex(re.compile(self.url))
            self.log(f" --> Dropping {counts} URLs from {docs} documents.", 2)

        if self.drop_html:
            counts, docs = drop_regex(re.compile(self.html))
            self.log(f" --> Dropping {counts} HTML tags from {docs} documents.", 2)

        if self.drop_emojis:
            counts, docs = drop_regex(re.compile(self.emojis))
            self.log(f" --> Dropping {counts} emojis from {docs} documents.", 2)

        if self.drop_numbers:
            counts, docs = drop_regex(re.compile(self.numbers))
            self.log(f" --> Dropping {counts} numbers from {docs} documents.", 2)

        if self.drop_punctuation:
            func = lambda row: row.translate(str.maketrans("", "", self.punctuation))
            X[corpus] = X[corpus].apply(func)
            self.log(f" --> Dropping punctuation from the text.", 2)

        return X


class Tokenizer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Tokenize the corpus.

    Convert documents into sequences of words. Additionally, create
    bigrams or trigrams (represented by words united with underscores,
    e.g. "New_York"). The transformations are applied on the column
    named `Corpus`. If there is no column with that name, an exception
    is raised.

    Parameters
    ----------
    bigram_freq: int, float or None, optional (default=None)
        Frequency threshold for bigram creation.
            - If None: Don't create any bigrams.
            - If int: Minimum number of occurrences to make a bigram.
            - If float: Minimum frequency fraction to make a bigram.

    trigram_freq: int, float or None, optional (default=None)
        Frequency threshold for trigram creation.
            - If None: Don't create any trigrams.
            - If int: Minimum number of occurrences to make a trigram.
            - If float: Minimum frequency fraction to make a trigram.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, Logger or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    Attributes
    ----------
    bigrams: pd.DataFrame
        Created bigrams and their frequencies.

    trigrams: pd.DataFrame
        Created trigrams and their frequencies.

    """

    @typechecked
    def __init__(
        self,
        bigram_freq: Optional[SCALAR] = None,
        trigram_freq: Optional[SCALAR] = None,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.bigram_freq = bigram_freq
        self.trigram_freq = trigram_freq

        self.bigrams = pd.DataFrame(columns=["bigram", "frequency"])
        self.trigrams = pd.DataFrame(columns=["trigram", "frequency"])

    def transform(self, X, y=None):
        """Tokenize the text.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features). If X is
            not a pd.DataFrame, it should be composed of a single
            feature containing the text documents. Each document is
            expected to be a string.

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        X: pd.DataFrame
            Transformed corpus.

        """
        def replace_ngrams(row, ngram, sep="<&&>"):
            """Replace a ngram with one word unified by underscores."""
            row = "&>" + sep.join(row) + "<&"  # Indicate words with separator
            row = row.replace(  # Replace ngrams' separator with underscore
                "&>" + sep.join(ngram) + "<&",
                "&>" + "_".join(ngram) + "<&",
            )
            return row[2:-2].split(sep)

        X, y = self._prepare_input(X, y)
        corpus = get_corpus(X)

        self.log("Tokenizing the corpus...", 1)

        try:  # Download tokenizer if not already on machine
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
        X[corpus] = X[corpus].apply(lambda row: nltk.word_tokenize(row))

        if self.bigram_freq:
            # Search for all bigrams in the documents
            finder = BigramCollocationFinder.from_documents(X[corpus])

            if self.bigram_freq < 1:
                self.bigram_freq = int(self.bigram_freq * len(finder.ngram_fd))

            n_bigrams, counts = 0, 0
            for bigram, freq in finder.ngram_fd.items():
                if freq >= self.bigram_freq:
                    n_bigrams += 1
                    counts += freq
                    X[corpus] = X[corpus].apply(replace_ngrams, args=(bigram,))
                    self.bigrams = self.bigrams.append(
                        {"bigram": bigram, "frequency": freq}, ignore_index=True
                    )

            self.log(f" --> Creating {n_bigrams} bigrams on {counts} locations.", 2)

        if self.trigram_freq:
            # Search for all trigrams in the documents
            finder = TrigramCollocationFinder.from_documents(X[corpus])

            if self.trigram_freq < 1:
                self.trigram_freq = int(self.trigram_freq * len(finder.ngram_fd))

            n_trigrams, counts = 0, 0
            for trigram, freq in finder.ngram_fd.items():
                if freq >= self.trigram_freq:
                    n_trigrams += 1
                    counts += freq
                    X[corpus] = X[corpus].apply(replace_ngrams, args=(trigram,))
                    self.trigrams = self.trigrams.append(
                        {"trigram": trigram, "frequency": freq}, ignore_index=True
                    )

            self.log(f" --> Creating {n_trigrams} trigrams on {counts} locations.", 2)

        return X


class Normalizer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Normalize the corpus.

    The transformations are applied on the column named `Corpus`,
    in the same order the parameters are presented. If there is
    no column with that name, an exception is raised.

    Parameters
    ----------
    stopwords: str or sequence, optional (default="english")
        Sequence of words to remove from the text. If str, choose
        from one of the languages available in the nltk package.

    stem: bool, optional (default=True)
        Whether to apply stemming.

    lemmatize: bool, optional (default=True)
        Whether to apply lemmatization.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, Logger or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    """

    @typechecked
    def __init__(
        self,
        stopwords: Union[str, SEQUENCE_TYPES] = "english",
        stem: bool = True,
        lemmatize: bool = True,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.stopwords = stopwords
        self.stem = stem
        self.lemmatize = lemmatize

    def transform(self, X, y=None):
        """Normalize the text.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features). If X is
            not a pd.DataFrame, it should be composed of a single
            feature containing the text documents. Each document is
            expected to be a sequence of words. If they are strings,
            words are separated by spaces.

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        X: pd.DataFrame
            Transformed corpus.

        """
        def pos(tag):
            """Get part of speech from a tag."""
            if tag in ["JJ", "JJR", "JJS"]:
                return wordnet.ADJ
            elif tag in ["RB", "RBR", "RBS"]:
                return wordnet.ADV
            elif tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
                return wordnet.VERB
            else:  # "NN", "NNS", "NNP", "NNPS"
                return wordnet.NOUN

        X, y = self._prepare_input(X, y)
        corpus = get_corpus(X)

        self.log("Normalizing the corpus...", 1)

        # If the corpus is not tokenized, separate by space
        if isinstance(X[corpus][0], str):
            X[corpus] = X[corpus].apply(lambda row: row.split())

        if self.stopwords:
            # Get stopwords from the NLTK library
            if isinstance(self.stopwords, str):
                try:  # Download resource if not already on machine
                    nltk.data.find("corpora/stopwords")
                except LookupError:
                    nltk.download("stopwords")
                self.stopwords = set(nltk.corpus.stopwords.words(self.stopwords))

            self.log(f" --> Dropping stopwords from the text.", 2)
            f = lambda row: [word for word in row if word not in self.stopwords]
            X[corpus] = X[corpus].apply(f)

        if self.stem:
            self.log(f" --> Applying stemming to the words.", 2)
            ps = PorterStemmer()
            X[corpus] = X[corpus].apply(lambda row: [ps.stem(word) for word in row])

        if self.lemmatize:
            try:  # Download resource if not already on machine
                nltk.data.find("corpora/wordnet")
            except LookupError:
                nltk.download("wordnet")
            try:
                nltk.data.find("taggers")
            except LookupError:
                nltk.download("averaged_perceptron_tagger")

            self.log(f" --> Applying lemmatization to the words.", 2)
            wnl = WordNetLemmatizer()
            f = lambda row: [wnl.lemmatize(w, pos(tag)) for w, tag in nltk.pos_tag(row)]
            X[corpus] = X[corpus].apply(f)

        return X


class Vectorizer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Vectorize text data.

    Transform the corpus into meaningful vectors of numbers. The
    transformation is applied on the column named `Corpus`. If
    there is no column with that name, an exception is raised.

    Parameters
    ----------
    strategy: str, optional (default="BOW")
        Strategy with which to vectorize the text. Available
        options are:
            - "BOW": Uses a Bag of Words algorithm.
            - "TF-IDF": Uses a TF-IDF algorithm.
            - "Hashing": Uses a hashing algorithm.

    verbose: int, optional (default=0)
        Verbosity level of the class. Possible values are:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, Logger or None, optional (default=None)
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    <strategy>: sklearn estimator
        Estimator instance (lowercase strategy) used to vectorize the
        corpus, e.g. `vectorizer.tfidf` for the TF-IDF strategy.

    """

    @typechecked
    def __init__(
        self,
        strategy: str = "BOW",
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
        **kwargs,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.strategy = strategy
        self.kwargs = kwargs

        self._estimator = None
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit to data.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features). If X is
            not a pd.DataFrame, it should be composed of a single
            feature containing the text documents. Each document
            can either be a string or a sequence of words.

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        X: pd.DataFrame
            Transformed corpus.

        """
        X, y = self._prepare_input(X, y)
        corpus = get_corpus(X)

        # Convert sequence of tokens to space separated string
        if not isinstance(X[corpus][0], str):
            X[corpus] = X[corpus].apply(lambda row: " ".join(row))

        if self.strategy.lower() == "bow":
            self.bow = self._estimator = CountVectorizer(**self.kwargs)
        elif self.strategy.lower() in ("tfidf", "tf-idf"):
            self.tfidf = self._estimator = TfidfVectorizer(**self.kwargs)
        elif self.strategy.lower() == "hashing":
            self.hashing = self._estimator = HashingVectorizer(**self.kwargs)
        else:
            raise ValueError(
                "Invalid value for the strategy parameter, got "
                f"{self.strategy}. Choose from: BOW, TF-IDF, Hashing."
            )

        self._estimator.fit(X[corpus])

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Vectorize the text.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features). If X is
            not a pd.DataFrame, it should be composed of a single
            feature containing the text documents. Each document
            can either be a string or a sequence of words.

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        X: pd.DataFrame
            Transformed corpus.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)
        corpus = get_corpus(X)

        self.log("Vectorizing the corpus...", 1)

        # Convert sequence of tokens to space separated string
        if not isinstance(X[corpus][0], str):
            X[corpus] = X[corpus].apply(lambda row: " ".join(row))

        matrix = self._estimator.transform(X[corpus]).toarray()
        if self.strategy.lower() != "hashing":
            for i, word in enumerate(self._estimator.get_feature_names()):
                X[word] = matrix[:, i]
        else:
            # Hashing has no words to put as column names
            for i, word in enumerate(range(matrix.shape[1])):
                X[f"hash_{i}"] = matrix[:, i]

        return X.drop(corpus, axis=1)
