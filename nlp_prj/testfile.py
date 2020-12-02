import time
from sklearn.metrics import accuracy_score
import pickle
import nltk
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
import nltk
nltk.download('punkt')
twenty_train = fetch_20newsgroups(subset='train', shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True, random_state=42)
X_train = twenty_train.data
y_train = twenty_train.target
X_test = twenty_test.data
y_test = twenty_test.target
def train(classifier,X_train,y_train,X_test,y_test):
    start = time.time()

    classifier.fit(X_train, y_train)
    end = time.time()
    predicted = classifier.predict(X_test)

    print("Accuracy: ", accuracy_score(y_test,predicted))
    print("Time duration: " + str(end - start))
    return classifier


def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]

model_2 = Pipeline([ ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer, stop_words=stopwords.words('english') + list(string.punctuation))), ('classifier', MultinomialNB(alpha=0.05))])
model_2 = train(model_2, X_train,y_train,X_test,y_test)