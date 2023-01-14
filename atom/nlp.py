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

import nltk
import pandas as pd
from nltk.collocations import (
    BigramCollocationFinder, QuadgramCollocationFinder,
    TrigramCollocationFinder,
)
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.base import BaseEstimator
from typeguard import typechecked

from atom.basetransformer import BaseTransformer
from atom.data_cleaning import TransformerMixin
from atom.utils import (
    DATAFRAME, FEATURES, INT, SCALAR, SEQUENCE, TARGET, CustomDict,
    check_is_fitted, composed, crash, get_corpus, is_sparse, merge,
    method_to_log, to_df,
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
        >>> from sklearn.datasets import fetch_20newsgroups

        >>> X, y = fetch_20newsgroups(
        ...     return_X_y=True,
        ...     categories=[
        ...         'alt.atheism',
        ...         'sci.med',
        ...         'comp.windows.x',
        ...     ],
        ...     shuffle=True,
        ...     random_state=1,
        ... )
        >>> X = np.array(X).reshape(-1, 1)

        >>> atom = ATOMClassifier(X, y)
        >>> print(atom.dataset)

        0     From: thssjxy@iitmax.iit.edu (Smile) Subject:...       2
        1     From: nancyo@fraser.sfu.ca (Nancy Patricia O'C...       0
        2     From: beck@irzr17.inf.tu-dresden.de (Andre Bec...       1
        3     From: keith@cco.caltech.edu (Keith Allan Schne...       0
        4     From: strom@Watson.Ibm.Com (Rob Strom) Subjec...       0
                                                         ...     ...
        2841  From: dreitman@oregon.uoregon.edu (Daniel R. R...       3
        2842  From: ethan@cs.columbia.edu (Ethan Solomita) ...       1
        2843  From: r0506048@cml3 (Chun-Hung Lin) Subject: ...       1
        2844  From: eshneken@ux4.cso.uiuc.edu (Edward A Shne...       2
        2845  From: ibeshir@nyx.cs.du.edu (Ibrahim) Subject...       2

        [2846 rows x 2 columns]

        >>> atom.textclean(verbose=2)

        Fitting TextCleaner...
        Cleaning the corpus...
         --> Decoding unicode characters to ascii.
         --> Converting text to lower case.
         --> Dropping 10012 emails from 2830 documents.
         --> Dropping 0 URL links from 0 documents.
         --> Dropping 2214 HTML tags from 1304 documents.
         --> Dropping 2 emojis from 1 documents.
         --> Dropping 31222 numbers from 2843 documents.
         --> Dropping punctuation from the text.

        >>> print(atom.dataset)

                                                        corpus  target
        0     from  smile subject forsale   used guitar amp...       2
        1     from  nancy patricia oconnor subject re amusi...       0
        2     from  andre beck subject re animation with xp...       1
        3     from  keith allan schneider subject re moralt...       0
        4     from  rob strom subject re socmotss et al pri...       0
                                                         ...     ...
        2841  from  daniel r reitman attorney to be subject...       3
        2842  from  ethan solomita subject forcing a window...       1
        2843  from r0506048cml3 chunhung lin subject re xma...       1
        2844  from  edward a shnekendorf subject airline ti...       2
        2845  from  ibrahim subject terminal for sale orga...       2

        [2846 rows x 2 columns]

        ```

    === "stand-alone"
        ```pycon
        >>> from atom.nlp import TextCleaner
        >>> from sklearn.datasets import fetch_20newsgroups

        >>> X, y = fetch_20newsgroups(
        ...     return_X_y=True,
        ...     categories=[
        ...         'alt.atheism',
        ...         'sci.med',
        ...         'comp.windows.x',
        ...     ],
        ...     shuffle=True,
        ...     random_state=1,
        ... )
        >>> X = np.array(X).reshape(-1, 1)

        >>> textcleaner = TextCleaner(verbose=2)
        >>> X = textcleaner.transform(X)

        Cleaning the corpus...
         --> Decoding unicode characters to ascii.
         --> Converting text to lower case.
         --> Dropping 10012 emails from 2830 documents.
         --> Dropping 0 URL links from 0 documents.
         --> Dropping 2214 HTML tags from 1304 documents.
         --> Dropping 2 emojis from 1 documents.
         --> Dropping 31222 numbers from 2843 documents.
         --> Dropping punctuation from the text.

        >>> print(X)

                                                         corpus
        0     from donald mackie  subject re barbecued food...
        1     from  david stockton subject re krillean phot...
        2     from  julia miller subject posix message cata...
        3     from   subject re yet more rushdie re islamic...
        4     from  joseph a muller subject jfk autograph f...
                                                         ...
        2841  from  joel reymont subject motif maling list\...
        2842  from  daniel paul checkman subject re is msg ...
        2843  from  ad absurdum per aspera subject re its a...
        2844  from  ralf subject items for sale organizati...
        2845  from  walter g seefeld subject klipsch kg1 sp...

        [2846 rows x 1 columns]

        ```

    """

    _train_only = False

    @typechecked
    def __init__(
        self,
        *,
        decode: bool = True,
        lower_case: bool = True,
        drop_email: bool = True,
        regex_email: str | None = None,
        drop_url: bool = True,
        regex_url: str | None = None,
        drop_html: bool = True,
        regex_html: str | None = None,
        drop_emoji: bool = True,
        regex_emoji: str | None = None,
        drop_number: bool = True,
        regex_number: str | None = None,
        drop_punctuation: bool = True,
        verbose: INT = 0,
        logger: str | Logger | None = None,
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
    def transform(self, X: FEATURES, y: TARGET | None = None) -> DATAFRAME:
        """Apply the transformations to the data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a dataframe, it should be composed of a single feature
            containing the text documents.

        y: int, str, sequence, dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

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
                elem.encode("ASCII", errors="strict")  # Returns bytes object
            except UnicodeEncodeError:
                norm = unicodedata.normalize("NFKD", elem)
                return "".join([c for c in norm if not unicodedata.combining(c)])
            else:
                return elem  # Return unchanged if encoding was successful

        def drop_regex(search: str) -> tuple[int, int]:
            """Find and remove a regex expression from the text.

            Parameters
            ----------
            search: str
                Regex pattern to search for.

            Returns
            -------
            int
                Number of occurrences.

            int
                Number of documents (rows) with occurrences.

            """
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
            if isinstance(X[corpus].iat[0], str):
                X[corpus] = X[corpus].apply(lambda elem: to_ascii(elem))
            else:
                X[corpus] = X[corpus].apply(lambda elem: [to_ascii(str(w)) for w in elem])
        self.log(" --> Decoding unicode characters to ascii.", 2)

        if self.lower_case:
            if isinstance(X[corpus].iat[0], str):
                X[corpus] = X[corpus].str.lower()
            else:
                X[corpus] = X[corpus].apply(lambda elem: [str(w).lower() for w in elem])
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
            if isinstance(X[corpus].iat[0], str):
                func = lambda row: row.translate(trans_table)
            else:
                func = lambda row: [str(w).translate(trans_table) for w in row]
            X[corpus] = X[corpus].apply(func)
            self.log(" --> Dropping punctuation from the text.", 2)

        # Convert all drops to one dataframe attribute
        self.drops = pd.concat(drops.values(), axis=1)

        # Drop empty tokens from every row
        if not isinstance(X[corpus].iat[0], str):
            X[corpus] = X[corpus].apply(lambda row: [w for w in row if w])

        return X


class TextNormalizer(BaseEstimator, TransformerMixin, BaseTransformer):
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
        - If None: Doesn't save a logging file.
        - If str: Name of the log file. Use "auto" for automatic naming.
        - Else: Python `logging.Logger` instance.

    See Also
    --------
    atom.nlp:TextCleaner
    atom.nlp:Tokenizer
    atom.nlp:Vectorizer

    Examples
    --------

    === "atom"
        ```pycon
        >>> from atom import ATOMClassifier

        >>> X = [
        ...    ["I àm in ne'w york"],
        ...    ["New york is nice"],
        ...    ["new york"],
        ...    ["hi there this is a test!"],
        ...    ["another line..."],
        ...    ["new york is larger than washington"],
        ...    ["running the test"],
        ...    ["this is a test"],
        ... ]
        >>> y = [1, 0, 0, 1, 1, 1, 0, 0]

        >>> atom = ATOMClassifier(X, y)
        >>> print(atom.dataset)

                                       corpus  target
        0                    running the test       0
        1            hi there this is a test!       1
        2                      this is a test       0
        3  new york is larger than washington       1
        4                    New york is nice       0
        5                   I àm in ne'w york       1
        6                     another line...       1
        7                            new york       0

        >>> atom.textnormalize(stopwords="english", lemmatize=True, verbose=2)

        Fitting TextNormalizer...
        Normalizing the corpus...
         --> Dropping stopwords.
         --> Applying lemmatization.

        >>> print(atom.dataset)

                                   corpus  target
        0                     [run, test]       0
        1                     [hi, test!]       1
        2                          [test]       0
        3  [new, york, large, washington]       1
        4               [New, york, nice]       0
        5             [I, àm, ne'w, york]       1
        6              [another, line...]       1
        7                     [new, york]       0

        ```

    === "stand-alone"
        ```pycon
        >>> from atom.nlp import TextNormalizer

        >>> X = [
        ...    ["I àm in ne'w york"],
        ...    ["New york is nice"],
        ...    ["new york"],
        ...    ["hi there this is a test!"],
        ...    ["another line..."],
        ...    ["new york is larger than washington"],
        ...    ["running the test"],
        ...    ["this is a test"],
        ... ]
        >>> y = [1, 0, 0, 1, 1, 1, 0, 0]

        >>> textnormalizer = TextNormalizer(
        ...     stopwords="english",
        ...     lemmatize=True,
        ...     verbose=2,
        ... )
        >>> X = textnormalizer.transform(X)

        Fitting TextNormalizer...
        Normalizing the corpus...
         --> Dropping stopwords.
         --> Applying lemmatization.

        >>> print(X)

                                   corpus
        0             [I, àm, ne'w, york]
        1               [New, york, nice]
        2                     [new, york]
        3                     [hi, test!]
        4              [another, line...]
        5  [new, york, large, washington]
        6                     [run, test]
        7                          [test]

        ```

    """

    _train_only = False

    @typechecked
    def __init__(
        self,
        *,
        stopwords: bool | str = True,
        custom_stopwords: SEQUENCE | None = None,
        stem: bool | str = False,
        lemmatize: bool = True,
        verbose: INT = 0,
        logger: str | Logger | None = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.stopwords = stopwords
        self.custom_stopwords = custom_stopwords
        self.stem = stem
        self.lemmatize = lemmatize

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: FEATURES, y: TARGET | None = None) -> DATAFRAME:
        """Normalize the text.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a dataframe, it should be composed of a single feature
            containing the text documents.

        y: int, str, sequence, dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

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

        X, y = self._prepare_input(X, y)
        corpus = get_corpus(X)

        self.log("Normalizing the corpus...", 1)

        # If the corpus is not tokenized, separate by space
        if isinstance(X[corpus].iat[0], str):
            X[corpus] = X[corpus].apply(lambda row: row.split())

        stopwords = []
        if self.stopwords:
            if self.stopwords is True:
                self.stopwords = "english"

            # Get stopwords from the NLTK library
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
            self.log(" --> Applying lemmatization.", 2)
            wnl = WordNetLemmatizer()
            f = lambda row: [wnl.lemmatize(w, pos(tag)) for w, tag in nltk.pos_tag(row)]
            X[corpus] = X[corpus].apply(f)

        return X


class Tokenizer(BaseEstimator, TransformerMixin, BaseTransformer):
    """Tokenize the corpus.

    Convert documents into sequences of words. Additionally,
    create n-grams (represented by words united with underscores,
    e.g. "New_York") based on their frequency in the corpus. The
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

    See Also
    --------
    atom.nlp:TextCleaner
    atom.nlp:TextNormalizer
    atom.nlp:Vectorizer

    Examples
    --------

    === "atom"
        ```pycon
        >>> from atom import ATOMClassifier

        >>> X = [
        ...    ["I àm in ne'w york"],
        ...    ["New york is nice"],
        ...    ["new york"],
        ...    ["hi there this is a test!"],
        ...    ["another line..."],
        ...    ["new york is larger than washington"],
        ...    ["running the test"],
        ...    ["this is a test"],
        ... ]
        >>> y = [1, 0, 0, 1, 1, 1, 0, 0]

        >>> atom = ATOMClassifier(X, y)
        >>> print(atom.dataset)

                                       corpus  target
        0                            new york       0
        1  new york is larger than washington       1
        2                    New york is nice       0
        3                   I àm in ne'w york       1
        4                      this is a test       0
        5                     another line...       1
        6                    running the test       0
        7            hi there this is a test!       1

        >>> atom.tokenize(verbose=2)

        Fitting Tokenizer...
        Tokenizing the corpus...

        >>> print(atom.dataset)

                                              corpus  target
        0                                [new, york]       0
        1  [new, york, is, larger, than, washington]       1
        2                      [New, york, is, nice]       0
        3                [I, àm, in, ne, ', w, york]       1
        4                        [this, is, a, test]       0
        5                       [another, line, ...]       1
        6                       [running, the, test]       0
        7          [hi, there, this, is, a, test, !]       1

        ```

    === "stand-alone"
        ```pycon
        >>> from atom.nlp import Tokenizer

        >>> X = [
        ...    ["I àm in ne'w york"],
        ...    ["New york is nice"],
        ...    ["new york"],
        ...    ["hi there this is a test!"],
        ...    ["another line..."],
        ...    ["new york is larger than washington"],
        ...    ["running the test"],
        ...    ["this is a test"],
        ... ]
        >>> y = [1, 0, 0, 1, 1, 1, 0, 0]

        >>> tokenizer = Tokenizer(bigram_freq=2, verbose=2)
        >>> X = tokenizer.transform(X)

        Fitting Tokenizer...
        Tokenizing the corpus...
         --> Creating 5 bigrams on 10 locations.

        >>> print(X)

                                             corpus
        0               [I, àm, in, ne, ', w, york]
        1                      [New, york_is, nice]
        2                                [new_york]
        3           [hi, there, this_is, a_test, !]
        4                      [another, line, ...]
        5  [new, york_is, larger, than, washington]
        6                      [running, the, test]
        7                         [this_is, a_test]

        ```

    """

    _train_only = False

    @typechecked
    def __init__(
        self,
        bigram_freq: SCALAR | None = None,
        trigram_freq: SCALAR | None = None,
        quadgram_freq: SCALAR | None = None,
        *,
        verbose: INT = 0,
        logger: str | Logger | None = None,
    ):
        super().__init__(verbose=verbose, logger=logger)
        self.bigram_freq = bigram_freq
        self.trigram_freq = trigram_freq
        self.quadgram_freq = quadgram_freq

        self.bigrams = None
        self.trigrams = None
        self.quadgrams = None

    @composed(crash, method_to_log, typechecked)
    def transform(self, X: FEATURES, y: TARGET | None = None) -> DATAFRAME:
        """Tokenize the text.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a dataframe, it should be composed of a single feature
            containing the text documents.

        y: int, str, sequence, dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

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
                Document in the corpus.

            ngram: tuple of str
                Words in the ngram.

            Returns
            -------
            str
               Document in the corpus with unified ngrams.

            """
            sep = "<&&>"  # Separator between words in a ngram.

            row = "&>" + sep.join(row) + "<&"  # Indicate words with separator
            row = row.replace(  # Replace ngrams separator with underscore
                "&>" + sep.join(ngram) + "<&",
                "&>" + "_".join(ngram) + "<&",
            )

            return row[2:-2].split(sep)

        X, y = self._prepare_input(X, y)
        corpus = get_corpus(X)

        self.log("Tokenizing the corpus...", 1)

        if isinstance(X[corpus].iat[0], str):
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

                if rows:
                    # Sort ngrams by frequency and add the dataframe as attribute
                    df = pd.DataFrame(rows).sort_values("frequency", ascending=False)
                    setattr(self, attr, df.reset_index(drop=True))

                    self.log(f" --> Creating {occur} {attr} on {counts} locations.", 2)
                else:
                    self.log(f" --> No {attr} found in the corpus.")

        return X


class Vectorizer(BaseEstimator, TransformerMixin, BaseTransformer):
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
        Device on which to train the estimators. Use any string
        that follows the [SYCL_DEVICE_FILTER][] filter selector,
        e.g. `device="gpu"` to use the GPU. Read more in the
        [user guide][accelerating-pipelines].

    engine: str, default="sklearn"
        Execution engine to use for the estimators. Refer to the
        [user guide][accelerating-pipelines] for an explanation
        regarding every choice. Choose from:

        - "sklearn" (only if device="cpu")
        - "cuml" (only if device="gpu")

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
        corpus, e.g. `vectorizer.tfidf` for the tfidf strategy.

    feature_names_in_: np.array
        Names of features seen during fit.

    n_features_in_: int
        Number of features seen during fit.


    See Also
    --------
    atom.nlp:TextCleaner
    atom.nlp:TextNormalizer
    atom.nlp:Tokenizer

    Examples
    --------

    === "atom"
        ```pycon
        >>> from atom import ATOMClassifier

        >>> X = [
        ...    ["I àm in ne'w york"],
        ...    ["New york is nice"],
        ...    ["new york"],
        ...    ["hi there this is a test!"],
        ...    ["another line..."],
        ...    ["new york is larger than washington"],
        ...    ["running the test"],
        ...    ["this is a test"],
        ... ]
        >>> y = [1, 0, 0, 1, 1, 1, 0, 0]

        >>> atom = ATOMClassifier(X, y)
        >>> print(atom.dataset)

                                       corpus  target
        0                            new york       0
        1                   I àm in ne'w york       1
        2                      this is a test       0
        3                    running the test       0
        4                     another line...       1
        5            hi there this is a test!       1
        6                    New york is nice       0
        7  new york is larger than washington       1

        >>> atom.vectorize(strategy="tfidf", verbose=2)

        Fitting Vectorizer...
        Vectorizing the corpus...

        >>> print(atom.dataset)

           corpus_another  corpus_in  corpus_is  ...  corpus_york  corpus_àm  target
        0        0.000000   0.000000   0.000000  ...     0.627914   0.000000       0
        1        0.000000   0.523358   0.000000  ...     0.422242   0.523358       1
        2        0.000000   0.000000   0.614189  ...     0.000000   0.000000       0
        3        0.000000   0.000000   0.000000  ...     0.000000   0.000000       0
        4        0.707107   0.000000   0.000000  ...     0.000000   0.000000       1
        5        0.000000   0.000000   0.614189  ...     0.000000   0.000000       1
        6        0.000000   0.000000   0.614189  ...     0.495524   0.000000       0
        7        0.000000   0.000000   0.614189  ...     0.495524   0.000000       1
        [8 rows x 13 columns]

        ```

    === "stand-alone"
        ```pycon
        >>> from atom.nlp import Vectorizer

        >>> X = [
        ...    ["I àm in ne'w york"],
        ...    ["New york is nice"],
        ...    ["new york"],
        ...    ["hi there this is a test!"],
        ...    ["another line..."],
        ...    ["new york is larger than washington"],
        ...    ["running the test"],
        ...    ["this is a test"],
        ... ]
        >>> y = [1, 0, 0, 1, 1, 1, 0, 0]

        >>> vectorizer = Vectorizer(strategy="tfidf", verbose=2)
        >>> X = vectorizer.fit_transform(X)

        Fitting Vectorizer...
        Vectorizing the corpus...

        >>> print(X)

           corpus_another  corpus_hi  ...  corpus_york  corpus_àm
        0        0.000000   0.000000  ...     0.343774   0.542162
        1        0.000000   0.000000  ...     0.415657   0.000000
        2        0.000000   0.000000  ...     0.659262   0.000000
        3        0.000000   0.525049  ...     0.000000   0.000000
        4        0.707107   0.000000  ...     0.000000   0.000000
        5        0.000000   0.000000  ...     0.304821   0.000000
        6        0.000000   0.000000  ...     0.000000   0.000000
        7        0.000000   0.000000  ...     0.000000   0.000000

        [8 rows x 18 columns]

        ```

    """

    _train_only = False

    @typechecked
    def __init__(
        self,
        strategy: str = "bow",
        *,
        return_sparse: bool = True,
        device: str = "cpu",
        engine: str = "sklearn",
        verbose: INT = 0,
        logger: str | Logger | None = None,
        **kwargs,
    ):
        super().__init__(device=device, engine=engine, verbose=verbose, logger=logger)
        self.strategy = strategy
        self.return_sparse = return_sparse
        self.kwargs = kwargs

        self._estimator = None
        self._is_fitted = False

    @composed(crash, method_to_log, typechecked)
    def fit(self, X: FEATURES, y: TARGET | None = None) -> Vectorizer:
        """Fit to data.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a dataframe, it should be composed of a single feature
            containing the text documents.

        y: int, str, sequence, dataframe-like or None, default=None
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
        if not isinstance(X[corpus].iat[0], str):
            X[corpus] = X[corpus].apply(lambda row: " ".join(row))

        strategies = CustomDict(
            bow="CountVectorizer",
            tfidf="TfidfVectorizer",
            hashing="HashingVectorizer",
        )

        if self.strategy in strategies:
            estimator = self._get_est_class(
                name=strategies[self.strategy],
                module="feature_extraction.text",
            )
            self._estimator = estimator(**self.kwargs)
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
    def transform(self, X: FEATURES, y: TARGET | None = None) -> DATAFRAME:
        """Vectorize the text.

        Parameters
        ----------
        X: dataframe-like
            Feature set with shape=(n_samples, n_features). If X is
            not a dataframe, it should be composed of a single feature
            containing the text documents.

        y: int, str, sequence, dataframe-like or None, default=None
            Does nothing. Implemented for continuity of the API.

        Returns
        -------
        dataframe
            Transformed corpus.

        """
        check_is_fitted(self)
        X, y = self._prepare_input(X, y)
        corpus = get_corpus(X)

        self.log("Vectorizing the corpus...", 1)

        # Convert sequence of tokens to space separated string
        if not isinstance(X[corpus].iat[0], str):
            X[corpus] = X[corpus].apply(lambda row: " ".join(row))

        matrix = self._estimator.transform(X[corpus])
        if hasattr(self._estimator, "get_feature_names_out"):
            columns = [f"corpus_{w}" for w in self._estimator.get_feature_names_out()]
        else:
            # Hashing has no words to put as column names
            columns = [f"hash{i}" for i in range(1, matrix.shape[1] + 1)]

        X = X.drop(corpus, axis=1)  # Drop original corpus column

        if "sklearn" not in self._estimator.__class__.__module__:
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

        return merge(X, to_df(matrix, X.index, columns))
