import pickle
import os
import math
import numpy as np
import pandas as pd
import nltk
import sys
import pickle
from sklearn.svm import LinearSVC
from nltk.stem import PorterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import string
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB

def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]
def classifier_naive(text):
    model_path = "C:\\Users\\Admin\\PycharmProjects\\nlp_p\\mysite4\\nlp_prj\\pickle_file\\navie.pickle"
    model = pickle.load(open(model_path,'rb'))
    return model.predict_proba(text).reshape(-1)
str = "From: v064mb9k@ubvmsd.cc.buffalo.edu (NEIL B. GANDLER)\nSubject: Need info on 88-89 Bonneville\nOrganization: University at Buffalo\nLines: 10\nNews-Software: VAX/VMS VNEWS 1.41\nNntp-Posting-Host: ubvmsd.cc.buffalo.edu\n\n\n I am a little confused on all of the models of the 88-89 bonnevilles."
arr =[]
arr.append(str)
def stemming_tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(w) for w in word_tokenize(text)]
model_1 = Pipeline([('vectorizer', TfidfVectorizer(stop_words=stopwords.words('english') + list(string.punctuation))), ('classifier', LinearSVC())])
model_2 = Pipeline([ ('vectorizer', TfidfVectorizer(tokenizer=stemming_tokenizer, stop_words=stopwords.words('english') + list(string.punctuation))), ('classifier', MultinomialNB(alpha=0.05))])


path = os.path.join(os.getcwd() ,"nlp_prj","pickle_file")
map_table = os.path.join(os.getcwd(),"nlp_prj")


def truncate(number, digits) -> float:
        stepper = 10.0 ** digits
        return math.trunc(stepper * number) / stepper

def map_labels_featues(key=[],per=[]):
        colnames = ['catergories', 'labels']
        #path = "C:\\Users\\Admin\\PycharmProjects\\nlp_p\\mysite4\\nlp_prj\\train.map"
        goal_dir = os.path.join(map_table, "train.map")

        print(goal_dir)
        data = pd.read_csv(goal_dir, names=colnames, delimiter=" ")
        print(key)
        # print("data :",data)
        # print("value: ",value)
        # print("goal dir",goal_dir)
        # print("data.loc[data['labels'] ==value].values.tolist()",data.loc[data['labels'] ==value].values.tolist())
        temp = {'top_1_accuracy': 'none',
                'top_2_accuracy': 'none',
                'top_3_accuracy': 'none',
                'top_4_accuracy': 'none',
                'top_1_percentage': 'none',
                'top_2_percentage': 'none',
                'top_3_percentage': 'none',
                'top_4_percentage': 'none',
                }

        for index in range(0, 4):
            try:
                if ('top_{}_accuracy'.format(index + 1) in temp):
                    #print(data.loc[data['labels'] == key[index]].values.reshape(-1)[0])
                    temp['top_{}_accuracy'.format(index + 1)] = (data.loc[data['labels'] == key[index]].values.reshape(-1)[0])
                    temp['top_{}_percentage'.format(index + 1)] = per[index]*100
            except:
                raise Exception("Not in top {} accuracy")
        return temp

def naive_bayes(value=[]):
        print("self.path ",self.path)
        naive_bayes_path = os.path.join(path, "naive_bayes.pickle")
        with open(naive_bayes_path, 'rb') as f:
            classifier = pickle.load(f)
            return classifier.predict(value)
def svm(self,value=[]):
        svm_path = os.path.join(path, "SVM.pickle")
        with open(svm_path, 'rb') as f:
            classifier = pickle.load(f)
            return classifier.predict(value)

def stemming_tokenizer(text):
        stemmer = PorterStemmer()
        return [stemmer.stem(w) for w in word_tokenize(text)]

def naive_bayes_probabilities(value=[],model_name="Naive"):
        model_path = os.path.join(path, "naive_bayes.pickle")
        if model_name=="Naive":
            model_path = os.path.join(path, "naive_bayes.pickle")
        elif model_name=="SVM":
            model_path = os.path.join(path, "SVC_model.pickle")
        elif model_name == "Grid":
            model_path = os.path.join(path, "navie.pickle")
        elif model_name == "Tree":
            model_path = os.path.join(path, "naive_bayes.pickle")
        elif model_name == "Linear":
            model_path = os.path.join(path, "linear.pickle")
        elif model_name == "Kneast":
            model_path = os.path.join(path, "naive_bayes.pickle")
        elif model_name == "DL1":
            model_path = os.path.join(path, "naive_bayes.pickle")
        elif model_name == "DL2":
            model_path = os.path.join(path, "naive_bayes.pickle")
        elif model_name == "More_1":
            model_path = os.path.join(path, "naive_bayes.pickle")
        else:
            model_path = os.path.join(path, "naive_bayes.pickle")
        #'C:\\Users\\Admin\\PycharmProjects\\nlp_p\\mysite4\\nlp_prj\\pickle_file\\naive_bayes.pickle'
        def stemming_tokenizer(text):
            stemmer = PorterStemmer()
            return [stemmer.stem(w) for w in word_tokenize(text)]
        with open(model_path, 'rb') as f:
            print("in")
            #classifier = pickle.load(f)
            result_probabilities2 = classifier_naive(value)
            #features_tagret_map = classifier.classes_
            #result_probabilities2 = classifier.predict_proba(value).reshape(-1)
            sorted_result, sorted_tagret =zip(*sorted(zip(result_probabilities2[::-1], features_tagret_map[::-1])))
            truncate_function = np.vectorize(truncate)
            sorted_result = truncate_function(sorted_result, 2)
            return map_labels_featues(list(sorted_tagret)[::-1][:4:1],list(sorted_result)[::-1][:4:1])

# p =Prediction()
# print(p.naive_bayes_probabilities(arr))