import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
# import mediapipe as mp
from keras.models import Sequential
from keras.layers import LSTM, Dense

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from sklearn import metrics
import pandas as pd
import numpy as np
import string
import nltk
from datasetsHF import load_dataset
import datasetsHF as ds

import uuid

import tensorflow as tf
# from object_detection.utils import config_util
# from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

from deep_translator import GoogleTranslator

from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

import re

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download(
    ["punkt", "wordnet", "omw-1.4", "averaged_perceptron_tagger", "universal_tagset"]
)

def text_translate(input):
    translated_text = GoogleTranslator(source='auto', target='en').translate(input)
    return translated_text

def text_preprocessing(text):
    stopwords = set()
    with open("static/en_stopwords.txt", "r") as file:
        for word in file:
            stopwords.add(word.rstrip("\n"))
    lemmatizer = WordNetLemmatizer()
    try:
        url_pattern = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
        user_pattern = r"@[^\s]+"
        entity_pattern = r"&.*;"
        neg_contraction = r"n't\W"
        non_alpha = "[^a-z]"
        cleaned_text = text.lower()
        cleaned_text = re.sub(neg_contraction, " not ", cleaned_text)
        cleaned_text = re.sub(url_pattern, " ", cleaned_text)
        cleaned_text = re.sub(user_pattern, " ", cleaned_text)
        cleaned_text = re.sub(entity_pattern, " ", cleaned_text)
        cleaned_text = re.sub(non_alpha, " ", cleaned_text)
        tokens = word_tokenize(cleaned_text)
        # provide POS tag for lemmatization to yield better result
        word_tag_tuples = pos_tag(tokens, tagset="universal")
        tag_dict = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
        final_tokens = []
        for word, tag in word_tag_tuples:
            if len(word) > 1 and word not in stopwords:
                if tag in tag_dict:
                    final_tokens.append(lemmatizer.lemmatize(word, tag_dict[tag]))
                else:
                    final_tokens.append(lemmatizer.lemmatize(word))
        return " ".join(final_tokens)
    except:
        return np.nan

def PredictSentiment(input):
    sentence_input = []
    # translated_text = text_translate(input)
    cleaned_text = text_translate(input)
    sentence_input.append(cleaned_text)

    # dtSet =  ds.load_dataset('Sp1786/multiclass-sentiment-analysis-dataset', split='train')
    # df = dtSet.to_pandas()
    df = pd.read_csv('cleaned_sped_sentiment_dataset_final.csv')
    df = df.dropna()

    stopset = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=list(stopset))

    y = df.values[:, 5]
    X = vectorizer.fit_transform(df.values[:, 3])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1000)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    test_predict_array = np.array(sentence_input)
    test_predict_vector = vectorizer.transform(test_predict_array)

    predicted_output = clf.predict(test_predict_vector)
    predicted_proba = clf.predict_proba(test_predict_vector)

    converted_proba = str(predicted_proba.tolist()).strip('[]')

    # print(probability_output)    
    print(converted_proba)
    return predicted_output[0], converted_proba

