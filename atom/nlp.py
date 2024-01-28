"""Automated Tool for Optimized Modeling (ATOM).

Author: Mavs
Description: Module containing the NLP transformers.

"""

from __future__ import annotations

import re
import unicodedata
from logging import Logger
from pathlib import Path
from string import punctuation

import nltk
import pandas as pd
from beartype import beartype
from nltk.collocations import (
    BigramCollocationFinder, QuadgramCollocationFinder,
    TrigramCollocationFinder,
)
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.base import OneToOneFeatureMixin
from sklearn.utils._set_output import _SetOutputMixin
from sklearn.utils.validation import _check_feature_names_in
from typing_extensions import Self

from atom.data_cleaning import TransformerMixin
from atom.utils.types import (
    Bool, DataFrame, Engine, FloatLargerZero, Pandas, Sequence,
    VectorizerStarts, Verbose, bool_t,
)
from atom.utils.utils import (
    check_is_fitted, check_nltk_module, composed, crash, get_corpus, is_sparse,
    merge, method_to_log, to_df,
)


@beartype
class TextCleaner(TransformerMixin, OneToOneFeatureMixin, _SetOutputMixin):
    r"""Applies standard text cleaning to the corpus.

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

    drop_number: bool, default=True
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
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    See Also
    --------
    atom.nlp:TextNormalizer
    atom.nlp:Tokenizer
    atom.nlp:Vectorizer

    Examples
    --------
    === "atom"
        ```pycon
        import numpy as np
        from atom import ATOMClassifier
        from sklearn.datasets import fetch_20newsgroups

        X, y = fetch_20newsgroups(
            return_X_y=True,
            categories=["alt.atheism", "sci.med", "comp.windows.x"],
            shuffle=True,
            random_state=1,
        )
        X = np.array(X).reshape(-1, 1)

        atom = ATOMClassifier(X, y, random_state=1)
        print(atom.dataset)

        atom.textclean(verbose=2)

        print(atom.dataset)
        ```

    === "stand-alone"
        ```pycon
        import numpy as np
        from atom.nlp import TextCleaner
        from sklearn.datasets import fetch_20newsgroups

        X, y = fetch_20newsgroups(
            return_X_y=True,
            categories=["alt.atheism", "sci.med", "comp.windows.x"],
            shuffle=True,
            random_state=1,
        )
        X = np.array(X).reshape(-1, 1)

        textcleaner = TextCleaner(verbose=2)
        X = textcleaner.transform(X)

        print(X)
        ```

    """

    def __init__(
        self,
        *,
        decode: Bool = True,
        lower_case: Bool = True,
        drop_email: Bool = True,
        regex_email: str | None = None,
        drop_url: Bool = True,
        regex_url: str | None = None,
        drop_html: Bool = True,
        regex_html: str | None = None,
        drop_emoji: Bool = True,
        regex_emoji: str | None = None,
        drop_number: Bool = True,
        regex_number: str | None = None,
        drop_punctuation: Bool = True,
        verbose: Verbose = 0,
        logger: str | Path | Logger | None = None,
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

    @composed(crash, method_to_log)
    def transform(self, X: DataFrame, y: Pandas | None = None) -> DataFrame:
        """Apply the transformations to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a dataframe, it should be composed of a single feature
            containing the text documents.

        y: int, str, sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Transformed corpus.

        """

        def to_ascii(elem: str) -> str:
            """Convert unicode string to ascii.

            Parameters
            ----------
            elem: str
                Elements of the corpus.

            Returns
            -------
            str
                ASCII string.

            """
            try:
                elem.encode("ASCII", errors="strict")  # Returns byes object
            except UnicodeEncodeError:
                norm = unicodedata.normalize("NFKD", elem)
                return "".join([c for c in norm if not unicodedata.combining(c)])
            else:
                return elem  # Return unchanged if encoding was successful

        def drop_regex(regex: str):
            """Find and remove a regex expression from the corpus.

            Parameters
            ----------
            regex: str
                Regex pattern to replace.

            """
            if isinstance(X[corpus].iloc[0], str):
                X[corpus] = X[corpus].str.replace(regex, "", regex=True)
            else:
                X[corpus] = X[corpus].apply(lambda x: [re.sub(regex, "", w) for w in x])

        corpus = get_corpus(X)

        self._log("Cleaning the corpus...", 1)

        if self.decode:
            if isinstance(X[corpus].iloc[0], str):
                X[corpus] = X[corpus].apply(lambda x: to_ascii(x))
            else:
                X[corpus] = X[corpus].apply(lambda doc: [to_ascii(str(w)) for w in doc])
        self._log(" --> Decoding unicode characters to ascii.", 2)

        if self.lower_case:
            self._log(" --> Converting text to lower case.", 2)
            if isinstance(X[corpus].iloc[0], str):
                X[corpus] = X[corpus].str.lower()
            else:
                X[corpus] = X[corpus].apply(lambda doc: [str(w).lower() for w in doc])

        if self.drop_email:
            if not self.regex_email:
                self.regex_email = r"[\w.-]+@[\w-]+\.[\w.-]+"

            self._log(" --> Dropping emails from documents.", 2)
            drop_regex(self.regex_email)

        if self.drop_url:
            if not self.regex_url:
                self.regex_url = r"https?://\S+|www\.\S+"

            self._log(" --> Dropping URL links from documents.", 2)
            drop_regex(self.regex_url)

        if self.drop_html:
            if not self.regex_html:
                self.regex_html = r"<.*?>"

            self._log(" --> Dropping HTML tags from documents.", 2)
            drop_regex(self.regex_html)

        if self.drop_emoji:
            if not self.regex_emoji:
                self.regex_emoji = r":[a-z_]+:"

            self._log(" --> Dropping emojis from documents.", 2)
            drop_regex(self.regex_emoji)

        if self.drop_number:
            if not self.regex_number:
                self.regex_number = r"\b\d+\b"

            self._log(" --> Dropping numbers from documents.", 2)
            drop_regex(self.regex_number)

        if self.drop_punctuation:
            self._log(" --> Dropping punctuation from the text.", 2)
            trans_table = str.maketrans("", "", punctuation)  # Translation table
            if isinstance(X[corpus].iloc[0], str):
                func = lambda doc: doc.translate(trans_table)
            else:
                func = lambda doc: [str(w).translate(trans_table) for w in doc]
            X[corpus] = X[corpus].apply(func)

        # Drop empty tokens from every document
        if not isinstance(X[corpus].iloc[0], str):
            X[corpus] = X[corpus].apply(lambda doc: [w for w in doc if w])

        return X


@beartype
class TextNormalizer(TransformerMixin, OneToOneFeatureMixin, _SetOutputMixin):
    """Normalize the corpus.

    Convert words to a more uniform standard. The transformations
    are applied on the column named `corpus`, in the same order the
    parameters are presented. If there is no column with that name,
    an exception is raised. If the provided documents are strings,
    words are separated by spaces.

    This class can be accessed from atom through the [textnormalize]
    [atomclassifier-textnormalize] method. Read more in the [user guide]
    [text-normalization].

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
        Whether to apply stemming using [SnowballStemmer][].

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
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    Attributes
    ----------
    feature_names_in_: np.ndarray
        Names of features seen during `fit`.

    n_features_in_: int
        Number of features seen during `fit`.

    See Also
    --------
    atom.nlp:TextCleaner
    atom.nlp:Tokenizer
    atom.nlp:Vectorizer

    Examples
    --------
    === "atom"
        ```pycon
        from atom import ATOMClassifier

        X = [
           ["I àm in ne'w york"],
           ["New york is nice"],
           ["new york"],
           ["hi there this is a test!"],
           ["another line..."],
           ["new york is larger than washington"],
           ["running the test"],
           ["this is a test"],
        ]
        y = [1, 0, 0, 1, 1, 1, 0, 0]

        atom = ATOMClassifier(X, y, test_size=2, random_state=1)
        print(atom.dataset)

        atom.textnormalize(stopwords="english", lemmatize=True, verbose=2)

        print(atom.dataset)
        ```

    === "stand-alone"
        ```pycon
        from atom.nlp import TextNormalizer

        X = [
           ["I àm in ne'w york"],
           ["New york is nice"],
           ["new york"],
           ["hi there this is a test!"],
           ["another line..."],
           ["new york is larger than washington"],
           ["running the test"],
           ["this is a test"],
        ]

        textnormalizer = TextNormalizer(
            stopwords="english",
            lemmatize=True,
            verbose=2,
        )
        X = textnormalizer.transform(X)

        print(X)
        ```

    """

    def __init__(
        self,
        *,
        stopwords: Bool | str = True,
        custom_stopwords: Sequence[str] | None = None,
        stem: Bool | str = False,
        lemmatize: Bool = True,
        verbose: Verbose = 0,
        logger: str | Path | Logger | None = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.stopwords = stopwords
        self.custom_stopwords = custom_stopwords
        self.stem = stem
        self.lemmatize = lemmatize

    @composed(crash, method_to_log)
    def transform(self, X: DataFrame, y: Pandas | None = None) -> DataFrame:
        """Normalize the text.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a dataframe, it should be composed of a single feature
            containing the text documents.

        y: int, str, sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Transformed corpus.

        """

        def pos(tag: str) -> wordnet.ADJ | wordnet.ADV | wordnet.VERB | wordnet.NOUN:
            """Get part of speech from a tag.

            Parameters
            ----------
            tag: str
                Wordnet tag corresponding to a word.

            Returns
            -------
            ADJ, ADV, VERB or NOUN
                Part of speech of word.

            """
            if tag in ("JJ", "JJR", "JJS"):
                return wordnet.ADJ
            elif tag in ("RB", "RBR", "RBS"):
                return wordnet.ADV
            elif tag in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
                return wordnet.VERB
            else:  # "NN", "NNS", "NNP", "NNPS"
                return wordnet.NOUN

        corpus = get_corpus(X)

        self._log("Normalizing the corpus...", 1)

        # If the corpus is not tokenized, separate by space
        if isinstance(X[corpus].iloc[0], str):
            X[corpus] = X[corpus].apply(lambda row: row.split())

        stopwords = set()
        if self.stopwords:
            if isinstance(self.stopwords, bool_t):
                self.stopwords = "english"

            # Get stopwords from the NLTK library
            check_nltk_module("corpora/stopwords", quiet=self.verbose < 2)
            stopwords = set(nltk.corpus.stopwords.words(self.stopwords.lower()))

        # Join predefined with customs stopwords
        if self.custom_stopwords is not None:
            stopwords = stopwords | set(self.custom_stopwords)

        if stopwords:
            self._log(" --> Dropping stopwords.", 2)
            f = lambda row: [word for word in row if word not in stopwords]
            X[corpus] = X[corpus].apply(f)

        if self.stem:
            if isinstance(self.stem, bool_t):
                self.stem = "english"

            self._log(" --> Applying stemming.", 2)
            ss = SnowballStemmer(language=self.stem.lower())
            X[corpus] = X[corpus].apply(lambda row: [ss.stem(word) for word in row])

        if self.lemmatize:
            self._log(" --> Applying lemmatization.", 2)
            check_nltk_module("corpora/wordnet", quiet=self.verbose < 2)
            check_nltk_module("taggers/averaged_perceptron_tagger", quiet=self.verbose < 2)
            check_nltk_module("corpora/omw-1.4", quiet=self.verbose < 2)

            wnl = WordNetLemmatizer()
            f = lambda row: [wnl.lemmatize(w, pos(tag)) for w, tag in nltk.pos_tag(row)]
            X[corpus] = X[corpus].apply(f)

        return X


@beartype
class Tokenizer(TransformerMixin, OneToOneFeatureMixin, _SetOutputMixin):
    """Tokenize the corpus.

    Convert documents into sequences of words. Additionally,
    create n-grams (represented by words united with underscores,
    e.g., "New_York") based on their frequency in the corpus. The
    transformations are applied on the column named `corpus`. If
    there is no column with that name, an exception is raised.

    This class can be accessed from atom through the [tokenize]
    [atomclassifier-tokenize] method. Read more in the [user guide]
    [tokenization].

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
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    Attributes
    ----------
    bigrams_: pd.DataFrame
        Created bigrams and their frequencies.

    trigrams_: pd.DataFrame
        Created trigrams and their frequencies.

    quadgrams_: pd.DataFrame
        Created quadgrams and their frequencies.

    feature_names_in_: np.ndarray
        Names of features seen during `fit`.

    n_features_in_: int
        Number of features seen during `fit`.

    See Also
    --------
    atom.nlp:TextCleaner
    atom.nlp:TextNormalizer
    atom.nlp:Vectorizer

    Examples
    --------
    === "atom"
        ```pycon
        from atom import ATOMClassifier

        X = [
           ["I àm in ne'w york"],
           ["New york is nice"],
           ["new york"],
           ["hi there this is a test!"],
           ["another line..."],
           ["new york is larger than washington"],
           ["running the test"],
           ["this is a test"],
        ]
        y = [1, 0, 0, 1, 1, 1, 0, 0]

        atom = ATOMClassifier(X, y, test_size=2, random_state=1)
        print(atom.dataset)

        atom.tokenize(verbose=2)

        print(atom.dataset)
        ```

    === "stand-alone"
        ```pycon
        from atom.nlp import Tokenizer

        X = [
           ["I àm in ne'w york"],
           ["New york is nice"],
           ["new york"],
           ["hi there this is a test!"],
           ["another line..."],
           ["new york is larger than washington"],
           ["running the test"],
           ["this is a test"],
        ]

        tokenizer = Tokenizer(bigram_freq=2, verbose=2)
        X = tokenizer.transform(X)

        print(X)
        ```

    """

    def __init__(
        self,
        bigram_freq: FloatLargerZero | None = None,
        trigram_freq: FloatLargerZero | None = None,
        quadgram_freq: FloatLargerZero | None = None,
        *,
        verbose: Verbose = 0,
        logger: str | Path | Logger | None = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.bigram_freq = bigram_freq
        self.trigram_freq = trigram_freq
        self.quadgram_freq = quadgram_freq

    @composed(crash, method_to_log)
    def transform(self, X: DataFrame, y: Pandas | None = None) -> DataFrame:
        """Tokenize the text.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a dataframe, it should be composed of a single feature
            containing the text documents.

        y: int, str, sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Transformed corpus.

        """

        def replace_ngrams(row: list[str], ngram: tuple[str]) -> list[str]:
            """Replace a ngram with one word unified by underscores.

            Parameters
            ----------
            row: list of str
                A document in the corpus.

            ngram: tuple of str
                Words in the ngram.

            Returns
            -------
            str
               Document in the corpus with unified ngrams.

            """
            sep = "<&&>"  # Separator between words in a ngram.

            row_c = "&>" + sep.join(row) + "<&"  # Indicate words with separator
            row_c = row_c.replace(  # Replace ngrams separator with underscore
                "&>" + sep.join(ngram) + "<&",
                "&>" + "_".join(ngram) + "<&",
            )

            return row_c[2:-2].split(sep)

        corpus = get_corpus(X)

        self._log("Tokenizing the corpus...", 1)

        if isinstance(X[corpus].iloc[0], str):
            check_nltk_module("tokenizers/punkt", quiet=self.verbose < 2)
            X[corpus] = X[corpus].apply(lambda row: nltk.word_tokenize(row))

        ngrams = {
            "bigrams": BigramCollocationFinder,
            "trigrams": TrigramCollocationFinder,
            "quadgrams": QuadgramCollocationFinder,
        }

        for attr, finder in ngrams.items():
            if frequency := getattr(self, f"{attr[:-1]}_freq"):
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

                if rows:
                    # Sort ngrams by frequency and add the dataframe as attribute
                    df = pd.DataFrame(rows).sort_values("frequency", ascending=False)
                    setattr(self, f"{attr}_", df.reset_index(drop=True))

                    self._log(f" --> Creating {occur} {attr} on {counts} locations.", 2)
                else:
                    self._log(f" --> No {attr} found in the corpus.", 2)

        return X


@beartype
class Vectorizer(TransformerMixin, _SetOutputMixin):
    """Vectorize text data.

    Transform the corpus into meaningful vectors of numbers. The
    transformation is applied on the column named `corpus`. If
    there is no column with that name, an exception is raised.

    If strategy="bow" or "tfidf", the transformed columns are named
    after the word they are embedding with the prefix `corpus_`. If
    strategy="hashing", the columns are named hash[N], where N stands
    for the n-th hashed column.

    This class can be accessed from atom through the [vectorize]
    [atomclassifier-vectorize] method. Read more in the [user guide]
    [vectorization].

    Parameters
    ----------
    strategy: str, default="bow"
        Strategy with which to vectorize the text. Choose from:

        - "[bow][]": Bag of Words.
        - "[tfidf][]": Term Frequency - Inverse Document Frequency.
        - "[hashing][]": Vectorize to a matrix of token occurrences.

    return_sparse: bool, default=True
        Whether to return the transformation output as a dataframe
        of sparse arrays. Must be False when there are other columns
        in X (besides `corpus`) that are non-sparse.

    device: str, default="cpu"
        Device on which to run the estimators. Use any string that
        follows the [SYCL_DEVICE_FILTER][] filter selector, e.g.
        `#!python device="gpu"` to use the GPU. Read more in the
        [user guide][gpu-acceleration].

    engine: str, dict or None, default=None
        Execution engine to use for [data][data-acceleration] and
        [estimators][estimator-acceleration]. The value should be
        one of the possible values to change one of the two engines,
        or a dictionary with keys `data` and `estimator`, with their
        corresponding choice as values to change both engines. If
        None, the default values are used. Choose from:

        - "data":

            - "numpy" (default)
            - "pyarrow"
            - "modin"

        - "estimator":

            - "sklearn" (default)
            - "cuml"

    verbose: int, default=0
        Verbosity level of the class. Choose from:

        - 0 to not print anything.
        - 1 to print basic information.
        - 2 to print detailed information.

    logger: str, Logger or None, default=None
        - If None: Logging isn't used.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    **kwargs
        Additional keyword arguments for the `strategy` estimator.

    Attributes
    ----------
    [strategy]_: sklearn transformer
        Estimator instance (lowercase strategy) used to vectorize the
        corpus, e.g., `vectorizer.tfidf` for the tfidf strategy.

    feature_names_in_: np.ndarray
        Names of features seen during `fit`.

    n_features_in_: int
        Number of features seen during `fit`.


    See Also
    --------
    atom.nlp:TextCleaner
    atom.nlp:TextNormalizer
    atom.nlp:Tokenizer

    Examples
    --------
    === "atom"
        ```pycon
        from atom import ATOMClassifier

        X = [
           ["I àm in ne'w york"],
           ["New york is nice"],
           ["new york"],
           ["hi there this is a test!"],
           ["another line..."],
           ["new york is larger than washington"],
           ["running the test"],
           ["this is a test"],
        ]
        y = [1, 0, 0, 1, 1, 1, 0, 0]

        atom = ATOMClassifier(X, y, test_size=2, random_state=1)
        print(atom.dataset)

        atom.vectorize(strategy="tfidf", verbose=2)

        print(atom.dataset)
        ```

    === "stand-alone"
        ```pycon
        from atom.nlp import Vectorizer

        X = [
           ["I àm in ne'w york"],
           ["New york is nice"],
           ["new york"],
           ["hi there this is a test!"],
           ["another line..."],
           ["new york is larger than washington"],
           ["running the test"],
           ["this is a test"],
        ]

        vectorizer = Vectorizer(strategy="tfidf", verbose=2)
        X = vectorizer.fit_transform(X)

        print(X)
        ```

    """

    def __init__(
        self,
        strategy: VectorizerStarts = "bow",
        *,
        return_sparse: Bool = True,
        device: str = "cpu",
        engine: Engine = None,
        verbose: Verbose = 0,
        logger: str | Path | Logger | None = None,
        **kwargs,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose, logger=logger)
        self.strategy = strategy
        self.return_sparse = return_sparse
        self.kwargs = kwargs

    def _get_corpus_columns(self) -> list[str]:
        """Get the names of the columns created by the vectorizer.

        Returns
        -------
        list of str
            Column names.

        """
        if hasattr(self._estimator, "get_feature_names_out"):
            return [f"{self._corpus}_{w}" for w in self._estimator.get_feature_names_out()]
        elif hasattr(self._estimator, "get_feature_names"):
            # cuML estimators have a different method name (returns a cudf.Series)
            return [f"{self._corpus}_{w}" for w in self._estimator.get_feature_names().to_numpy()]
        else:
            raise ValueError(
                "The get_feature_names_out method is not available for strategy='hashing'."
            )

    @composed(crash, method_to_log)
    def fit(self, X: DataFrame, y: Pandas | None = None) -> Self:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a dataframe, it should be composed of a single feature
            containing the text documents.

        y: int, str, sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        Self
            Estimator instance.

        """
        self._corpus = get_corpus(X)

        # Convert a sequence of tokens to space separated string
        if not isinstance(X[self._corpus].iloc[0], str):
            X[self._corpus] = X[self._corpus].apply(lambda row: " ".join(row))

        strategies = {
            "bow": "CountVectorizer",
            "tfidf": "TfidfVectorizer",
            "hashing": "HashingVectorizer",
        }

        estimator = self._get_est_class(
            name=strategies[self.strategy],
            module="feature_extraction.text",
        )
        self._estimator = estimator(**self.kwargs)

        if hasattr(self._estimator, "set_output"):
            # transform="pandas" fails for sparse output
            self._estimator.set_output(transform="default")

        self._log("Fitting Vectorizer...", 1)
        self._estimator.fit(X[self._corpus])

        # Add the estimator as attribute to the instance
        setattr(self, f"{self.strategy}_", self._estimator)

        return self

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Get output feature names for transformation.

        Parameters
        ----------
        input_features: sequence or None, default=None
            Only used to validate feature names with the names seen in
            `fit`.

        Returns
        -------
        np.ndarray
            Transformed feature names.

        """
        check_is_fitted(self, attributes="feature_names_in_")
        _check_feature_names_in(self, input_features)

        og_columns = [c for c in self.feature_names_in_ if c != self._corpus]
        return og_columns + self._get_corpus_columns()

    @composed(crash, method_to_log)
    def transform(self, X: DataFrame, y: Pandas | None = None) -> DataFrame:
        """Vectorize the text.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a dataframe, it should be composed of a single feature
            containing the text documents.

        y: int, str, sequence, dataframe-like or None, default=None
            Do nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Transformed corpus.

        """
        self._log("Vectorizing the corpus...", 1)

        # Convert a sequence of tokens to space-separated string
        if not isinstance(X[self._corpus].iloc[0], str):
            X[self._corpus] = X[self._corpus].apply(lambda row: " ".join(row))

        matrix = self._estimator.transform(X[self._corpus])
        X = X.drop(columns=self._corpus)  # Drop original corpus column

        if "sklearn" not in self._estimator.__class__.__module__:
            matrix = matrix.get()  # Convert cupy sparse array back to scipy

        if not self.return_sparse:
            self._log(" --> Converting the output to a full array.", 2)
            matrix = matrix.toarray()
        elif not X.empty and not is_sparse(X):
            # Raise if there are other columns that are non-sparse
            raise ValueError(
                "Invalid value for the return_sparse parameter. The value must "
                "must be False when X contains non-sparse columns (besides corpus)."
            )

        if self.strategy != "hashing":
            columns = self._get_corpus_columns()
        else:
            # Hashing has no words to put as column names
            columns = [f"hash{i}" for i in range(1, matrix.shape[1] + 1)]

        return merge(X, to_df(matrix, index=X.index, columns=columns))
