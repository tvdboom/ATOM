# -*- coding: utf-8 -*-

"""
Automated Tool for Optimized Modelling (ATOM)
Author: Mavs
Description: Module containing the NLP transformers.

"""

from __future__ import annotations

import re
import unicodedata
from logging import Logger
from string import punctuation
from typing import Optional, Union

import nltk
import pandas as pd
from nltk.collocations import (
    BigramCollocationFinder, QuadgramCollocationFinder,
    TrigramCollocationFinder,
)
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import (
    CountVectorizer, HashingVectorizer, TfidfVectorizer,
)
from typeguard import typechecked

from atom.basetransformer import BaseTransformer
from atom.data_cleaning import TransformerMixin
from atom.utils import (
    INT, SCALAR, SEQUENCE_TYPES, X_TYPES, Y_TYPES, CustomDict, check_is_fitted,
    composed, crash, get_corpus, is_sparse, method_to_log, to_df,
)


class TextCleaner(BaseEstimator, TransformerMixin, BaseTransformer):
    """Applies standard text cleaning to the corpus.

    Transformations include normalizing characters and dropping
    noise from the text (emails, HTML tags, URLs, etc...). The
    transformations are applied on the column named `corpus`, in
    the same order the parameters are presented. If there is no
    column with that name, an exception is raised.

    This class can be accessed from atom through the [textclean]
    [atomclassifier-textclean] method. Read more in the [user guide]
    [text-cleaning].

    Parameters
    ----------
    decode: bool, default=True
        Whether to decode unicode characters to their ascii
        representations.

    lower_case: bool, default=True
        Whether to convert all characters to lower case.

    drop_email: bool, default=True
        Whether to drop email addresses from the text.

    regex_email: str, default=None
        Regex used to search for email addresses. If None, it uses
        `r"[\w.-]+@[\w-]+\.[\w.-]+"`.

    drop_url: bool, default=True
        Whether to drop URL links from the text.

    regex_url: str, default=None
        Regex used to search for URLs. If None, it uses
        `r"https?://\S+|www\.\S+"`.

    drop_html: bool, default=True
        Whether to drop HTML tags from the text. This option is
        particularly useful if the data was scraped from a website.

    regex_html: str, default=None
        Regex used to search for html tags. If None, it uses
        `r"<.*?>"`.

    drop_emoji: bool, default=True
        Whether to drop emojis from the text.

    regex_emoji: str, default=None
        Regex used to search for emojis. If None, it uses
        `r":[a-z_]+:"`.

    drop_number: bool, default=False
        Whether to drop numbers from the text.

    regex_number: str, default=None
        Regex used to search for numbers. If None, it uses
        `r"\b\d+\b".` Note that numbers adjacent to letters are
        not removed.

    drop_punctuation: bool, default=True
        Whether to drop punctuations from the text. Characters
        considered punctuation are `!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~`.

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    logger: str, Logger or None, default=None
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    Attributes
    ----------
    drops: pd.DataFrame
        Encountered regex matches. The row indices correspond to
        the document index from which the occurrence was dropped.

    See Also
    --------
    atom.nlp:TextNormalizer
    atom.nlp:Tokenizer
    atom.nlp:Vectorizer

    Examples
    --------

    === "atom"
        ```pycon
        >>> from atom import ATOMClassifier
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

        >>> atom = ATOMClassifier(X, y)
        >>> print(atom.dataset)

             mean radius  mean texture  ...  worst fractal dimension  target
        0          17.99         10.38  ...                  0.11890       0
        1          12.25         17.94  ...                  0.08132       1
        2          13.87         20.70  ...                  0.08492       1
        3          12.06         12.74  ...                  0.07898       1
        4          12.62         17.15  ...                  0.07330       1
        ..           ...           ...  ...                      ...     ...
        564        11.34         18.61  ...                  0.06783       1
        565        11.43         17.31  ...                  0.08096       1
        566        11.06         14.96  ...                  0.09080       1
        567        13.20         15.82  ...                  0.08385       1
        568        20.55         20.86  ...                  0.07569       0

        [569 rows x 31 columns]

        >>> atom.scale(verbose=2)

        Fitting Scaler...
        Scaling features...

        >>> # Note the reduced number of rows
        >>> print(atom.dataset)

             mean radius  mean texture  ...  worst fractal dimension  target
        0       1.052603     -2.089926  ...                 1.952598       0
        1      -0.529046     -0.336627  ...                -0.114004       1
        2      -0.082657      0.303467  ...                 0.083968       1
        3      -0.581401     -1.542600  ...                -0.242685       1
        4      -0.427093     -0.519842  ...                -0.555040       1
        ..           ...           ...  ...                      ...     ...
        564    -0.779796     -0.181242  ...                -0.855847       1
        565    -0.754996     -0.482735  ...                -0.133801       1
        566    -0.856949     -1.027742  ...                 0.407321       1
        567    -0.267275     -0.828293  ...                 0.025126       1
        568     1.758008      0.340573  ...                -0.423609       0

        [569 rows x 31 columns]

        ```

    === "stand-alone"
        ```pycon
        >>> from atom.data_cleaning import Scaler
        >>> from sklearn.datasets import load_breast_cancer

        >>> X, y = load_breast_cancer(return_X_y=True, as_frame=True)

             mean radius  mean texture  ...  worst symmetry  worst fractal dimension
        0          17.99         10.38  ...          0.4601                  0.11890
        1          20.57         17.77  ...          0.2750                  0.08902
        2          19.69         21.25  ...          0.3613                  0.08758
        3          11.42         20.38  ...          0.6638                  0.17300
        4          20.29         14.34  ...          0.2364                  0.07678
        ..           ...           ...  ...             ...                      ...
        564        21.56         22.39  ...          0.2060                  0.07115
        565        20.13         28.25  ...          0.2572                  0.06637
        566        16.60         28.08  ...          0.2218                  0.07820
        567        20.60         29.33  ...          0.4087                  0.12400
        568         7.76         24.54  ...          0.2871                  0.07039

        [569 rows x 30 columns]

        >>> scaler = Scaler(verbose=2)
        >>> X = scaler.fit_transform(X)

        Fitting Scaler...
        Scaling features...

        >>> # Note the reduced number of rows
        >>> print(X)

             mean radius  mean texture  ...  worst symmetry  worst fractal dimension
        0       1.097064     -2.073335  ...        2.750622                 1.937015
        1       1.829821     -0.353632  ...       -0.243890                 0.281190
        2       1.579888      0.456187  ...        1.152255                 0.201391
        3      -0.768909      0.253732  ...        6.046041                 4.935010
        4       1.750297     -1.151816  ...       -0.868353                -0.397100
        ..           ...           ...  ...             ...                      ...
        564     2.110995      0.721473  ...       -1.360158                -0.709091
        565     1.704854      2.085134  ...       -0.531855                -0.973978
        566     0.702284      2.045574  ...       -1.104549                -0.318409
        567     1.838341      2.336457  ...        1.919083                 2.219635
        568    -1.808401      1.221792  ...       -0.048138                -0.751207

        [569 rows x 30 columns]


        ```

    """

    @typechecked
    def __init__(
        self,
        *,
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
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
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
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None) -> pd.DataFrame:
        """Apply the transformations to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a pd.DataFrame, it should be composed of a single
            feature containing the text documents.

        y: int, str, dict, sequence or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        pd.DataFrame
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
            for i, row in X[corpus].items():
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

        self.log("Cleaning the corpus...", 1)

        if self.decode:
            if isinstance(X[corpus].iloc[0], str):
                X[corpus] = X[corpus].apply(lambda row: to_ascii(row))
            else:
                X[corpus] = X[corpus].apply(lambda row: [to_ascii(str(w)) for w in row])
        self.log(" --> Decoding unicode characters to ascii.", 2)

        if self.lower_case:
            if isinstance(X[corpus].iloc[0], str):
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
            if isinstance(X[corpus].iloc[0], str):
                func = lambda row: row.translate(trans_table)
            else:
                func = lambda row: [str(w).translate(trans_table) for w in row]
            X[corpus] = X[corpus].apply(func)
            self.log(" --> Dropping punctuation from the text.", 2)

        # Convert all drops to one dataframe attribute
        self.drops = pd.concat(drops.values(), axis=1)

        # Drop empty tokens from every row
        if not isinstance(X[corpus][0], str):
            X[corpus] = X[corpus].apply(lambda row: [w for w in row if w])

        return X


class Tokenizer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Tokenize the corpus.

    Convert documents into sequences of words. Additionally,
    create n-grams (represented by words united with underscores,
    e.g. "New_York") based on their frequency in the corpus. The
    transformations are applied on the column named `corpus`. If
    there is no column with that name, an exception is raised.

    Parameters
    ----------
    bigram_freq: int, float or None, default=None
        Frequency threshold for bigram creation.
            - If None: Don't create any bigrams.
            - If int: Minimum number of occurrences to make a bigram.
            - If float: Minimum frequency fraction to make a bigram.

    trigram_freq: int, float or None, default=None
        Frequency threshold for trigram creation.
            - If None: Don't create any trigrams.
            - If int: Minimum number of occurrences to make a trigram.
            - If float: Minimum frequency fraction to make a trigram.

    quadgram_freq: int, float or None, default=None
        Frequency threshold for quadgram creation.
            - If None: Don't create any quadgrams.
            - If int: Minimum number of occurrences to make a quadgram.
            - If float: Minimum frequency fraction to make a quadgram.

    verbose: int, default=0
        Verbosity level of the class. Choose from:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, Logger or None, default=None
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
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
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
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a pd.DataFrame, it should be composed of a single
            feature containing the text documents.

        y: int, str, dict, sequence or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        pd.DataFrame
            Transformed corpus.

        """

        def replace_ngrams(row, ngram, sep="<&&>"):
            """Replace a ngram with one word unified by underscores."""
            row = "&>" + sep.join(row) + "<&"  # Indicate words with separator
            row = row.replace(  # Replace ngrams separator with underscore
                "&>" + sep.join(ngram) + "<&",
                "&>" + "_".join(ngram) + "<&",
            )
            return row[2:-2].split(sep)

        X, y = self._prepare_input(X, y)
        corpus = get_corpus(X)

        self.log("Tokenizing the corpus...", 1)

        if isinstance(X[corpus].iloc[0], str):
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
                # Search for all n-grams in the corpus
                ngram_fd = finder.from_documents(X[corpus]).ngram_fd

                if frequency < 1:
                    frequency = int(frequency * len(ngram_fd))

                rows = []
                occur, counts = 0, 0
                for ngram, freq in ngram_fd.items():
                    if freq >= frequency:
                        occur += 1
                        counts += freq
                        X[corpus] = X[corpus].apply(replace_ngrams, args=(ngram,))
                        rows.append({attr[:-1]: "_".join(ngram), "frequency": freq})

                # Sort ngrams by frequency and add the dataframe as attribute
                df = pd.DataFrame(rows).sort_values("frequency", ascending=False)
                setattr(self, attr, df.reset_index(drop=True))

                self.log(f" --> Creating {occur} {attr} on {counts} locations.", 2)

        return X


class TextNormalizer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Normalize the corpus.

    Convert words to a more uniform standard. The transformations
    are applied on the column named `corpus`, in the same order the
    parameters are presented. If there is no column with that name,
    an exception is raised. If the provided documents are strings,
    words are separated by spaces.

    Parameters
    ----------
    stopwords: bool or str, default=True
        Whether to remove a predefined dictionary of stopwords.
            - If False: Don't remove any predefined stopwords.
            - If True: Drop predefined english stopwords from the text.
            - If str: Language from `nltk.corpus.stopwords.words`.

    custom_stopwords: sequence or None, default=None
        Custom stopwords to remove from the text.

    stem: bool or str, default=False
        Whether to apply stemming using SnowballStemmer.
            - If False: Don't apply stemming.
            - If True: Apply stemmer based on the english language.
            - If str: Language from `SnowballStemmer.languages`.

    lemmatize: bool, default=True
        Whether to apply lemmatization using WordNetLemmatizer.

    verbose: int, default=0
        Verbosity level of the class. Choose from:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, Logger or None, default=None
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
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
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
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a pd.DataFrame, it should be composed of a single
            feature containing the text documents.

        y: int, str, dict, sequence or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        pd.DataFrame
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
        if isinstance(X[corpus].iloc[0], str):
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
                nltk.data.find("taggers/averaged_perceptron_tagger")
            except LookupError:
                nltk.download("averaged_perceptron_tagger")
            try:
                nltk.data.find("corpora/omw-1.4")
            except LookupError:
                nltk.download("omw-1.4")

            self.log(" --> Applying lemmatization.", 2)
            wnl = WordNetLemmatizer()
            f = lambda row: [wnl.lemmatize(w, pos(tag)) for w, tag in nltk.pos_tag(row)]
            X[corpus] = X[corpus].apply(f)

        return X


class Vectorizer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Vectorize text data.

    Transform the corpus into meaningful vectors of numbers. The
    transformation is applied on the column named `corpus`. If
    there is no column with that name, an exception is raised. The
    transformed columns are named after the word they are embedding
    with the prefix `corpus_`.

    Parameters
    ----------
    strategy: str, default="bow"
        Strategy with which to vectorize the text. Choose from:
            - "bow": Bag of Words.
            - "tfidf": Term Frequency - Inverse Document Frequency.
            - "hashing": Vectorize to a matrix of token occurrences.

    return_sparse: bool, default=True
        Whether to return the transformation output as a dataframe
        of sparse arrays. Must be False when there are other columns
        in X (besides `corpus`) that are non-sparse.

    gpu: bool or str, default=False
        Train LabelEncoder on GPU. Only for
        `encode_target=True`.
            - If False: Always use CPU implementation.
            - If True: Use GPU implementation if possible.
            - If "force": Force GPU implementation.

    verbose: int, default=0
        Verbosity level of the class. Choose from:
            - 0 to not print anything.
            - 1 to print basic information.
            - 2 to print detailed information.

    logger: str, Logger or None, default=None
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    [strategy]: sklearn transformer
        Estimator instance (lowercase strategy) used to vectorize the
        corpus, e.g. `vectorizer.tfidf` for the TF-IDF strategy.

    feature_names_in_: np.array
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.

    """

    @typechecked
    def __init__(
        self,
        strategy: str = "bow",
        return_sparse: bool = True,
        gpu: Union[bool, str] = False,
        verbose: INT = 0,
        logger: Optional[Union[str, Logger]] = None,
        **kwargs,
    ):
        super().__init__(gpu=gpu, verbose=verbose, logger=logger)
        self.strategy = strategy
        self.return_sparse = return_sparse
        self.kwargs = kwargs

        self._estimator = None
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a pd.DataFrame, it should be composed of a single
            feature containing the text documents.

        y: int, str, dict, sequence or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        Vectorizer
            Estimator instance.

        """
        X, y = self._prepare_input(X, y)
        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        corpus = get_corpus(X)

        # Convert sequence of tokens to space separated string
        if not isinstance(X[corpus][0], str):
            X[corpus] = X[corpus].apply(lambda row: " ".join(row))

        strategies = CustomDict(
            bow=self._get_gpu(CountVectorizer, "cuml.feature_extraction.text"),
            tfidf=self._get_gpu(TfidfVectorizer, "cuml.feature_extraction.text"),
            hashing=self._get_gpu(HashingVectorizer, "cuml.feature_extraction.text"),
        )

        if self.strategy in strategies:
            self._estimator = strategies[self.strategy](**self.kwargs)
        else:
            raise ValueError(
                "Invalid value for the strategy parameter, got "
                f"{self.strategy}. Choose from: {', '.join(strategies)}."
            )

        self.log("Fitting Vectorizer...", 1)
        self._estimator.fit(X[corpus])

        # Add the estimator as attribute to the instance
        setattr(self, self.strategy.lower(), self._estimator)

        self._is_fitted = True
        return self

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: X_TYPES, y: Optional[Y_TYPES] = None):
        """Vectorize the text.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a pd.DataFrame, it should be composed of a single
            feature containing the text documents.

        y: int, str, dict, sequence or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        pd.DataFrame
            Transformed corpus.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)
        corpus = get_corpus(X)

        self.log("Vectorizing the corpus...", 1)

        # Convert sequence of tokens to space separated string
        if not isinstance(X[corpus].iloc[0], str):
            X[corpus] = X[corpus].apply(lambda row: " ".join(row))

        matrix = self._estimator.transform(X[corpus])
        if hasattr(self._estimator, "get_feature_names_out"):
            columns = [f"{corpus}_{w}" for w in self._estimator.get_feature_names_out()]
        else:
            # Hashing has no words to put as column names
            columns = [f"hash{i}" for i in range(matrix.shape[1])]

        X = X.drop(corpus, axis=1)  # Drop original corpus column

        if self.gpu:
            matrix = matrix.get()  # Convert cupy sparse array back to scipy

            # cuML estimators have a slightly different method name
            if hasattr(self._estimator, "get_feature_names"):
                vocabulary = self._estimator.get_feature_names()  # cudf.Series
                columns = [f"{corpus}_{w}" for w in vocabulary.to_numpy()]

        if not self.return_sparse:
            self.log(" --> Converting the output to a full array.", 2)
            matrix = matrix.toarray()
        elif not X.empty and not is_sparse(X):
            # Raise if there are other columns that are non-sparse
            raise ValueError(
                "Invalid value for the return_sparse parameter. The value must "
                "must be False when X contains non-sparse columns (besides corpus)."
            )

        return pd.concat([X, to_df(matrix, X.index, columns)], axis=1)
