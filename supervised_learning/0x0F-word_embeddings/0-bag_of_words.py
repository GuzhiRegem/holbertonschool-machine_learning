#!/usr/bin/env python3
"""
    module
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """ bag_of_words """
    vectorizer = CountVectorizer(vocabulary=vocab)
    X_train_counts = vectorizer.fit_transform(sentences)
    embeddings = X_train_counts.toarray()
    features = list(vectorizer.get_feature_names_out())
    return embeddings, features
