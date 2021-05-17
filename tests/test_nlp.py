# coding: utf-8

"""Automated Tool for Optimized Modelling (ATOM)

Author: Mavs
Description: Unit tests for nlp.py

"""

# Standard packages
import pytest

# Own modules
from atom.nlp import TextCleaner, Tokenizer, Normalizer, Vectorizer
from .utils import X_bin, X_text


# Test TextCleaner ================================================= >>

def test_corpus_is_not_present():
    """Assert that an error is raised when there is no corpus."""
    pytest.raises(ValueError, TextCleaner().transform, X_bin)


def test_decode():
    """Assert that unicode characters are converted to ascii."""
    assert TextCleaner().transform([["tést"]])["Corpus"][0] == "test"


def test_lower_case():
    """Assert that characters are converted to lower case."""
    assert TextCleaner().transform([["TEST"]])["Corpus"][0] == "test"


def test_drop_email():
    """Assert that email addresses are dropped from the text."""
    assert TextCleaner().transform([["test@webmail.com"]])["Corpus"][0] == ""


def test_drop_url():
    """Assert that URLs are dropped from the text."""
    assert TextCleaner().transform([["www.test.com"]])["Corpus"][0] == ""


def test_drop_html():
    """Assert that html tags are dropped from the text."""
    assert TextCleaner().transform([["<table>test</table>"]])["Corpus"][0] == "test"


def test_drop_emojis():
    """Assert that emojis are dropped from the text."""
    assert TextCleaner().transform([[":test_emoji:"]])["Corpus"][0] == ""


def test_drop_numbers():
    """Assert that numbers are dropped from the text."""
    assert TextCleaner().transform([["123,123.123"]])["Corpus"][0] == ""


def test_drop_punctuation():
    """Assert that punctuations are dropped from the text."""
    assert TextCleaner().transform([["'test!?"]])["Corpus"][0] == "test"


# Test Tokenizer =================================================== >>

def test_tokenization():
    """Assert that the corpus is tokenized."""
    assert Tokenizer().transform([["A test"]])["Corpus"][0] == ["A", "test"]


def test_bigrams():
    """Assert that bigrams are created."""
    tokenizer = Tokenizer(bigram_freq=0.5)
    assert tokenizer.transform([["a b a b"]])["Corpus"][0] == ["a_b", "a_b"]


def test_trigrams():
    """Assert that trigrams are created."""
    tokenizer = Tokenizer(trigram_freq=0.5)
    assert tokenizer.transform([["a b c a b c"]])["Corpus"][0] == ["a_b_c", "a_b_c"]


# Test Normalizer ================================================== >>

def test_normalizer_space_separation():
    """Assert that the corpus is separated by space if not tokenized."""
    assert Normalizer().transform([["b c"]])["Corpus"][0] == ["b", "c"]


def test_stopwords_custom():
    """Assert that custom stopwords are removed."""
    assert Normalizer(stopwords=["b"]).transform([["a b"]])["Corpus"][0] == ["a"]


def test_stemming():
    """Assert that the corpus is stemmed."""
    assert Normalizer().transform([["running"]])["Corpus"][0] == ["run"]


def test_lemmatization():
    """Assert that the corpus is lemmatized."""
    assert Normalizer().transform([["better"]])["Corpus"][0] == ["well"]


# Test Vectorizer ================================================== >>

def test_vectorizer_space_separation():
    """Assert that the corpus is separated by space if not tokenized."""
    assert "hi" in Vectorizer().fit_transform({"corpus": [["hi", "there"], ["hi"]]})


def test_invalid_strategy():
    """Assert that an error is raised when the strategy is invalid."""
    vectorizer = Vectorizer(strategy="invalid")
    assert pytest.raises(ValueError, vectorizer.fit, X_text)


@pytest.mark.parametrize("strategy", ["bow", "tfidf", "tf-idf"])
def test_strategies(strategy):
    """Assert that the BOW and TF-IDF strategies works as intended."""
    X = Vectorizer(strategy=strategy).fit_transform(X_text)
    assert X.shape == (4, 10)
    assert "york" in X


def test_hashing():
    """Assert that the Hashing strategy works as intended."""
    X = Vectorizer(strategy="Hashing", n_features=10).fit_transform(X_text)
    assert X.shape == (4, 10)
    assert "hash_1" in X
