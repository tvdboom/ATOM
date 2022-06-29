# coding: utf-8

"""Automated Tool for Optimized Modelling (ATOM)

Author: Mavs
Description: Unit tests for nlp.py

"""

import pandas as pd
import pytest

from atom import ATOMClassifier
from atom.nlp import TextCleaner, TextNormalizer, Tokenizer, Vectorizer

from .conftest import X_bin, X_text, y10


# Test TextCleaner ================================================= >>

def test_corpus_is_not_present():
    """Assert that an error is raised when there is no corpus."""
    with pytest.raises(ValueError, match=r".*not contain a text corpus.*"):
        TextCleaner().transform(X_bin)


def test_decode():
    """Assert that unicode characters are converted to ascii."""
    assert TextCleaner().transform([["tést"]])["corpus"][0] == "test"


def test_lower_case():
    """Assert that characters are converted to lower case."""
    assert TextCleaner().transform([["TEST"]])["corpus"][0] == "test"


def test_drop_emails():
    """Assert that email addresses are dropped from the text."""
    cleaner = TextCleaner()
    assert cleaner.transform([["test@webmail.com"]])["corpus"][0] == ""
    assert not cleaner.drops["email"].dropna().empty


def test_drop_url():
    """Assert that URLs are dropped from the text."""
    cleaner = TextCleaner()
    assert cleaner.transform([["www.test.com"]])["corpus"][0] == ""
    assert not cleaner.drops["url"].dropna().empty


def test_drop_html():
    """Assert that html tags are dropped from the text."""
    cleaner = TextCleaner()
    assert cleaner.transform([["<table>test</table>"]])["corpus"][0] == "test"
    assert not cleaner.drops["html"].dropna().empty


def test_drop_emojis():
    """Assert that emojis are dropped from the text."""
    cleaner = TextCleaner()
    assert cleaner.transform([[":test_emoji:"]])["corpus"][0] == ""
    assert not cleaner.drops["emoji"].dropna().empty


def test_drop_numbers():
    """Assert that numbers are dropped from the text."""
    cleaner = TextCleaner()
    assert cleaner.transform([["123,123.123"]])["corpus"][0] == ""
    assert not cleaner.drops["number"].dropna().empty


def test_drop_punctuation():
    """Assert that punctuations are dropped from the text."""
    assert TextCleaner().transform([["'test!?"]])["corpus"][0] == "test"


def test_cleaner_tokenized():
    """Assert that the cleaner works for a tokenized corpus."""
    X = Tokenizer().transform(X_text)
    X = TextCleaner().transform(X)
    assert isinstance(X["corpus"][0], list)


def test_drop_empty_tokens():
    """Assert that empty tokens are dropped."""
    assert TextCleaner().transform([[[",;", "hi"]]])["corpus"][0] == ["hi"]


# Test Tokenizer =================================================== >>

def test_tokenization():
    """Assert that the corpus is tokenized."""
    X = Tokenizer().transform([["A test"]])
    assert X["corpus"][0] == ["A", "test"]


def test_bigrams():
    """Assert that bigrams are created."""
    tokenizer = Tokenizer(bigram_freq=0.5)
    X = tokenizer.transform([["a b a b"]])
    assert X["corpus"][0] == ["a_b", "a_b"]
    assert isinstance(tokenizer.bigrams, pd.DataFrame)


def test_trigrams():
    """Assert that trigrams are created."""
    tokenizer = Tokenizer(trigram_freq=0.5)
    X = tokenizer.transform([["a b c a b c"]])
    assert X["corpus"][0] == ["a_b_c", "a_b_c"]
    assert isinstance(tokenizer.trigrams, pd.DataFrame)


def test_quadgrams():
    """Assert that quadgrams are created."""
    tokenizer = Tokenizer(quadgram_freq=0.5)
    X = tokenizer.transform([["a b c d a b c d"]])
    assert X["corpus"][0] == ["a_b_c_d", "a_b_c_d"]
    assert isinstance(tokenizer.quadgrams, pd.DataFrame)


# Test TextNormalizer ================================================== >>

def test_normalizer_space_separation():
    """Assert that the corpus is separated by space if not tokenized."""
    assert TextNormalizer().transform([["b c"]])["corpus"][0] == ["b", "c"]


def test_stopwords():
    """Assert that predefined stopwords are removed."""
    assert TextNormalizer().transform([["a b"]])["corpus"][0] == ["b"]


def test_stopwords_custom():
    """Assert that custom stopwords are removed."""
    normalizer = TextNormalizer(stopwords=False, custom_stopwords=["b"])
    assert normalizer.transform([["a b"]])["corpus"][0] == ["a"]


def test_stemming():
    """Assert that the corpus is stemmed."""
    normalizer = TextNormalizer(stem=True, lemmatize=False)
    assert normalizer.transform([["running"]])["corpus"][0] == ["run"]


def test_lemmatization():
    """Assert that lemmatization is applied."""
    assert TextNormalizer().transform([["better"]])["corpus"][0] == ["well"]


# Test Vectorizer ================================================== >>

def test_vectorizer_space_separation():
    """Assert that the corpus is separated by space if not tokenized."""
    assert "corpus_hi" in Vectorizer().fit_transform({"corpus": [["hi"], ["hi"]]})


def test_invalid_strategy():
    """Assert that an error is raised when the strategy is invalid."""
    vectorizer = Vectorizer(strategy="invalid")
    with pytest.raises(ValueError, match=r".*value for the strategy.*"):
        vectorizer.fit(X_text)


@pytest.mark.parametrize("strategy", ["bow", "tfidf"])
def test_strategies(strategy):
    """Assert that the BOW and TF-IDF strategies works as intended."""
    X = Vectorizer(strategy=strategy).fit_transform(X_text)
    assert X.shape == (10, 20)
    assert "corpus_york" in X


def test_hashing():
    """Assert that the Hashing strategy works as intended."""
    X = Vectorizer(strategy="Hashing", n_features=10).fit_transform(X_text)
    assert X.shape == (10, 10)
    assert "hash_1" in X


def test_return_sparse():
    """Assert that the output is sparse."""
    X = Vectorizer(strategy="bow", return_sparse=True).fit_transform(X_text, y10)
    assert all(pd.api.types.is_sparse(X[c]) for c in X.columns)


def test_error_sparse_with_dense():
    """Assert that an error is raised when dense and sparse are combined."""
    atom = ATOMClassifier(X_text, y10, random_state=1)
    atom.apply(lambda x: 1, columns="new")  # Create dense column
    with pytest.raises(ValueError, match=r".*value for the return_sparse.*"):
        atom.vectorize(strategy="BOW", return_sparse=True)


def test_sparse_with_dense():
    """Assert that the output is dense when return_sparse=False."""
    atom = ATOMClassifier(X_text, y10, random_state=1)
    atom.apply(lambda x: 1, columns="new")  # Create dense column
    atom.vectorize(strategy="BOW", return_sparse=False)
    assert all(not pd.api.types.is_sparse(atom.X[c]) for c in atom.features)
