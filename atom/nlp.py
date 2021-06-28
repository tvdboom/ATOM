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
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.collocations import (
    BigramCollocationFinder,
    TrigramCollocationFinder,
    QuadgramCollocationFinder,
)

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

    regex_email: str, optional (default=None)
        Regex used to search for email addresses. If None, it uses
        `r"[\w.-]+@[\w-]+\.[\w.-]+"`.

    drop_url: bool, optional (default=True)
        Whether to drop URL links from the text.

    regex_url: str, optional (default=None)
        Regex used to search for URLs. If None, it uses
        `r"https?://\S+|www\.\S+"`.

    drop_html: bool, optional (default=True)
        Whether to drop HTML tags from the text. This option is
        particularly useful if the data was scraped from a website.

    regex_html: str, optional (default=None)
        Regex used to search for html tags. If None, it uses
        `r"<.*?>"`.

    drop_emoji: bool, optional (default=True)
        Whether to drop emojis from the text.

    regex_emoji: str, optional (default=None)
        Regex used to search for emojis. If None, it uses
        `r":[a-z_]+:"`.

    drop_number: bool, optional (default=False)
        Whether to drop numbers from the text.

    regex_number: str, optional (default=None)
        Regex used to search for numbers. If None, it uses
        `r"\b\d+\b".` Note that numbers adjacent to letters are
        not removed.

    drop_punctuation: bool, optional (default=True)
        Whether to drop punctuations from the text. Characters
        considered punctuation are `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`.

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
    drops: pd.DataFrame
        Encountered regex matches. The row indices correspond to
        the document index from which the occurrence was dropped.

    """

    @typechecked
    def __init__(
        self,
        decode: bool = True,
        lower_case: bool = True,
        drop_email: bool = True,
        regex_email: Optional[str] = None,
        drop_url: bool = True,
        regex_url: Optional[str] = None,
        drop_html: bool = True,
        regex_html: Optional[str] = None,
        drop_emoji: bool = True,
        regex_emoji: Optional[str] = None,
        drop_number: bool = True,
        regex_number: Optional[str] = None,
        drop_punctuation: bool = True,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.decode = decode
        self.lower_case = lower_case
        self.drop_email = drop_email
        self.regex_email = regex_email
        self.drop_url = drop_url
        self.regex_url = regex_url
        self.drop_html = drop_html
        self.regex_html = regex_html
        self.drop_emoji = drop_emoji
        self.regex_emoji = regex_emoji
        self.drop_number = drop_number
        self.regex_number = regex_number
        self.drop_punctuation = drop_punctuation

        # Encountered regex occurrences
        self.drops = pd.DataFrame()

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Apply the transformations to the data.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features). If X is
            not a pd.DataFrame, it should be composed of a single
            feature containing the text documents.

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
                row.encode("ASCII", errors="strict")  # Returns bytes object
            except UnicodeEncodeError:
                norm = unicodedata.normalize("NFKD", row)
                return "".join([c for c in norm if not unicodedata.combining(c)])
            else:
                return row  # Return unchanged if encoding was successful

        def drop_regex(search):
            """Find and remove a regex expression from the text."""
            counts, docs = 0, 0
            for i, row in enumerate(X[corpus]):
                for j, elem in enumerate([row] if isinstance(row, str) else row):
                    regex = getattr(self, f"regex_{search}")
                    occurrences = re.compile(regex).findall(elem)
                    if occurrences:
                        docs += 1
                        counts += len(occurrences)
                        drops[search].loc[i] = occurrences
                        for occ in occurrences:
                            if row is elem:
                                X[corpus][i] = X[corpus][i].replace(occ, "", 1)
                            else:
                                X[corpus][i][j] = X[corpus][i][j].replace(occ, "", 1)

            return counts, docs

        X, y = self._prepare_input(X, y)
        corpus = get_corpus(X)

        # Create a pd.Series for every type of drop
        drops = {}
        for elem in ("email", "url", "html", "emoji", "number"):
            drops[elem] = pd.Series(name=elem, dtype="object")

        self.log("Filtering the corpus...", 1)

        if self.decode:
            if isinstance(X[corpus][0], str):
                X[corpus] = X[corpus].apply(lambda row: to_ascii(row))
            else:
                X[corpus] = X[corpus].apply(lambda row: [to_ascii(str(w)) for w in row])
        self.log(" --> Decoding unicode characters to ascii.", 2)

        if self.lower_case:
            if isinstance(X[corpus][0], str):
                X[corpus] = X[corpus].str.lower()
            else:
                X[corpus] = X[corpus].apply(lambda row: [str(w).lower() for w in row])
        self.log(" --> Converting text to lower case.", 2)

        if self.drop_email:
            if not self.regex_email:
                self.regex_email = r"[\w.-]+@[\w-]+\.[\w.-]+"

            counts, docs = drop_regex("email")
            self.log(f" --> Dropping {counts} emails from {docs} documents.", 2)

        if self.drop_url:
            if not self.regex_url:
                self.regex_url = r"https?://\S+|www\.\S+"

            counts, docs = drop_regex("url")
            self.log(f" --> Dropping {counts} URL links from {docs} documents.", 2)

        if self.drop_html:
            if not self.regex_html:
                self.regex_html = r"<.*?>"

            counts, docs = drop_regex("html")
            self.log(f" --> Dropping {counts} HTML tags from {docs} documents.", 2)

        if self.drop_emoji:
            if not self.regex_emoji:
                self.regex_emoji = r":[a-z_]+:"

            counts, docs = drop_regex("emoji")
            self.log(f" --> Dropping {counts} emojis from {docs} documents.", 2)

        if self.drop_number:
            if not self.regex_number:
                self.regex_number = r"\b\d+\b"

            counts, docs = drop_regex("number")
            self.log(f" --> Dropping {counts} numbers from {docs} documents.", 2)

        if self.drop_punctuation:
            trans_table = str.maketrans("", "", punctuation)  # Translation table
            if isinstance(X[corpus][0], str):
                func = lambda row: row.translate(trans_table)
            else:
                func = lambda row: [str(w).translate(trans_table) for w in row]
            X[corpus] = X[corpus].apply(func)
            self.log(f" --> Dropping punctuation from the text.", 2)

        # Concatenate all drops to one dataframe attribute
        self.drops = pd.concat([series for series in drops.values()], axis=1)

        # Drop empty tokens from every row
        if not isinstance(X[corpus][0], str):
            X[corpus] = X[corpus].apply(lambda row: [w for w in row if w])

        return X


class Tokenizer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Tokenize the corpus.

    Convert documents into sequences of words. Additionally,
    create n-grams (represented by words united with underscores,
    e.g. "New_York") based on their frequency in the corpus. The
    transformations are applied on the column named `Corpus`. If
    there is no column with that name, an exception is raised.

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

    quadgram_freq: int, float or None, optional (default=None)
        Frequency threshold for quadgram creation.
            - If None: Don't create any quadgrams.
            - If int: Minimum number of occurrences to make a quadgram.
            - If float: Minimum frequency fraction to make a quadgram.

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

    quadgrams: pd.DataFrame
        Created quadgrams and their frequencies.

    """

    @typechecked
    def __init__(
        self,
        bigram_freq: Optional[SCALAR] = None,
        trigram_freq: Optional[SCALAR] = None,
        quadgram_freq: Optional[SCALAR] = None,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.bigram_freq = bigram_freq
        self.trigram_freq = trigram_freq
        self.quadgram_freq = quadgram_freq

        self.bigrams = None
        self.trigrams = None
        self.quadgrams = None

    def transform(self, X, y=None):
        """Tokenize the text.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features). If X is
            not a pd.DataFrame, it should be composed of a single
            feature containing the text documents.

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

        if isinstance(X[corpus][0], str):
            try:  # Download tokenizer if not already on machine
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt")
            X[corpus] = X[corpus].apply(lambda row: nltk.word_tokenize(row))

        ngrams = {
            "bigrams": BigramCollocationFinder,
            "trigrams": TrigramCollocationFinder,
            "quadgrams": QuadgramCollocationFinder,
        }

        for attr, finder in ngrams.items():
            frequency = getattr(self, f"{attr[:-1]}_freq")
            if frequency:
                df = pd.DataFrame(columns=[attr[:-1], "frequency"])

                # Search for all n-grams in the documents
                ngram_fd = finder.from_documents(X[corpus]).ngram_fd

                # Fraction to total number
                if frequency < 1:
                    frequency = int(frequency * len(ngram_fd))

                occur, counts = 0, 0
                for ngram, freq in ngram_fd.items():
                    if freq >= frequency:
                        occur += 1
                        counts += freq
                        X[corpus] = X[corpus].apply(replace_ngrams, args=(ngram,))
                        df = df.append(
                            {attr[:-1]: ngram, "frequency": freq}, ignore_index=True
                        )

                setattr(self, attr, df.sort_values(by="frequency", ascending=False))

                self.log(f" --> Creating {occur} {attr} on {counts} locations.", 2)

        return X


class Normalizer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Normalize the corpus.

    Convert words to a more uniform standard. The transformations
    are applied on the column named `Corpus`, in the same order the
    parameters are presented. If there is no column with that name,
    an exception is raised. If the provided documents are strings,
    words are separated by spaces.

    Parameters
    ----------
    stopwords: bool or str, optional (default=True)
        Whether to remove a predefined dictionary of stopwords.
            - If False: Don't remove any predefined stopwords.
            - If True: Drop predefined english stopwords from the text.
            - If str: Language from `nltk.corpus.stopwords.words`.

    custom_stopwords: sequence or None, optional (default=None)
        Custom stopwords to remove from the text.

    stem: bool or str, optional (default=False)
        Whether to apply stemming using SnowballStemmer.
            - If False: Don't apply stemming.
            - If True: Apply stemmer based on the english language.
            - If str: Language from `SnowballStemmer.languages`.

    lemmatize: bool, optional (default=True)
        Whether to apply lemmatization using WordNetLemmatizer.

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
        stopwords: Union[bool, str] = True,
        custom_stopwords: Optional[SEQUENCE_TYPES] = None,
        stem: Union[bool, str] = False,
        lemmatize: bool = True,
        verbose: int = 0,
        logger: Optional[Union[str, callable]] = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.stopwords = stopwords
        self.custom_stopwords = custom_stopwords
        self.stem = stem
        self.lemmatize = lemmatize

    def transform(self, X, y=None):
        """Normalize the text.

        Parameters
        ----------
        X: dict, list, tuple, np.ndarray or pd.DataFrame
            Feature set with shape=(n_samples, n_features). If X is
            not a pd.DataFrame, it should be composed of a single
            feature containing the text documents.

        y: int, str, sequence or None, optional (default=None)
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        X: pd.DataFrame
            Transformed corpus.

        """

        def pos(tag):
            """Get part of speech from a tag."""
            if tag in ("JJ", "JJR", "JJS"):
                return wordnet.ADJ
            elif tag in ("RB", "RBR", "RBS"):
                return wordnet.ADV
            elif tag in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
                return wordnet.VERB
            else:  # "NN", "NNS", "NNP", "NNPS"
                return wordnet.NOUN

        X, y = self._prepare_input(X, y)
        corpus = get_corpus(X)

        self.log("Normalizing the corpus...", 1)

        # If the corpus is not tokenized, separate by space
        if isinstance(X[corpus][0], str):
            X[corpus] = X[corpus].apply(lambda row: row.split())

        stopwords = []
        if self.stopwords:
            if self.stopwords is True:
                self.stopwords = "english"

            # Get stopwords from the NLTK library
            try:  # Download resource if not already on machine
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords")
            stopwords = list(set(nltk.corpus.stopwords.words(self.stopwords.lower())))

        # Join predefined with customs stopwords
        if self.custom_stopwords is not None:
            stopwords = set(stopwords + list(self.custom_stopwords))

        if stopwords:
            self.log(" --> Dropping stopwords.", 2)
            f = lambda row: [word for word in row if word not in stopwords]
            X[corpus] = X[corpus].apply(f)

        if self.stem:
            if self.stem is True:
                self.stem = "english"

            self.log(" --> Applying stemming.", 2)
            ss = SnowballStemmer(language=self.stem.lower())
            X[corpus] = X[corpus].apply(lambda row: [ss.stem(word) for word in row])

        if self.lemmatize:
            try:  # Download resource if not already on machine
                nltk.data.find("corpora/wordnet")
            except LookupError:
                nltk.download("wordnet")
            try:
                nltk.data.find("taggers")
            except LookupError:
                nltk.download("averaged_perceptron_tagger")

            self.log(" --> Applying lemmatization.", 2)
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
            feature containing the text documents.

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
            feature containing the text documents.

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
