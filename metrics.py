from datasetsHF import load_dataset, ClassLabel, load_metric
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import numpy as np

import pandas as pd

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download(
    ["punkt", "wordnet", "omw-1.4", "averaged_perceptron_tagger", "universal_tagset"]
)

from nltk.stem import WordNetLemmatizer
import re
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

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

# dataset = load_dataset("vibhorag101/suicide_prediction_dataset_phr")
df = pd.read_csv('cleaned_sped_sentiment_dataset_final.csv')
# dtSet =  load_dataset('Sp1786/multiclass-sentiment-analysis-dataset', split='train')
# dtSet = load_dataset('stanfordnlp/sst2', split='train')
# df = dtSet.to_pandas()
df = df.dropna()
df = df.drop(["Unnamed: 0"], axis=1)
# df["text"] = df["text"].apply(text_preprocessing)

stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=list(stopset))

y = df.values[:, 4]
X = vectorizer.fit_transform(df.values[:, 2])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1000)

clf = MultinomialNB()
clf.fit(X_train, y_train)


Y_pred = clf.predict(X_test)


acc = accuracy_score(y_test, Y_pred)
f1 = f1_score(y_test, Y_pred, average="weighted", pos_label="positive")
recall = recall_score(y_test, Y_pred, average="weighted", pos_label="positive")
precision = precision_score(y_test, Y_pred, average="weighted", pos_label="positive")
confusion = confusion_matrix(y_test, Y_pred)


print("Accuracy Score: " + str(acc))
print("F1 Score: " + str(f1))
print("Recall Score: " + str(recall))
print("Precision Score: " + str(precision))
print("Confusion Matrix: " + str(confusion))