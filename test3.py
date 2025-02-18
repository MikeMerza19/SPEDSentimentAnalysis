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

nltk.download('stopwords')

def PredictSentiment():
    sentence_input = []


    # translated_text = text_translate(input)

    sentence_input.append(input)

    dtSet =  ds.load_dataset('Sp1786/multiclass-sentiment-analysis-dataset', split='train')
    df = dtSet.to_pandas()
    df = df.dropna()

    dtSet.to_csv('C:/Users/WebDev/Desktop/SPED Sentiment Analysis v7.0/sped_sentiment_dataset.csv')

PredictSentiment()
#     stopset = set(stopwords.words('english'))
#     vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=list(stopset))

#     y = df.values[:, 3]
#     X = vectorizer.fit_transform(df.values[:, 1])

#     # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1000)

#     # clf = MultinomialNB()
#     # clf.fit(X_train, y_train)

#     # test_predict_array = np.array(sentence_input)
#     # test_predict_vector = vectorizer.transform(test_predict_array)

#     # predicted_output = clf.predict(test_predict_vector)
#     # predicted_proba = clf.predict_proba(test_predict_vector)

#     # converted_proba = str(predicted_proba.tolist()).strip('[]')

#     # # print(probability_output)
#     # # print(input)
#     # print(converted_proba)
#     # return predicted_output[0], converted_proba
#     print(X.shape)


# PredictSentiment("I love studying in SPED")