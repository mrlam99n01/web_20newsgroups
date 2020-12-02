from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import numpy as np
class NaiveBayes:
    def __init__(self):
        self.twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
        self.twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
        self.count_vect = CountVectorizer()
        self.X_train_counts = self.count_vect.fit_transform(self.twenty_train.data)
        self.tfidf_transformer = TfidfTransformer()
        self.X_train_tfidf = self.tfidf_transformer.fit_transform(self.X_train_counts)
        self.clf = MultinomialNB().fit(self.X_train_tfidf, self.twenty_train.target)
        self.text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', MultinomialNB())])
        self.text_clf = self.text_clf.fit(self.twenty_train.data, self.twenty_train.target)
    def predictedxz(self,data=[]):
        return self.text_clf.predict(data)