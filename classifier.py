from pyexpat import model
from numpy import vectorize
import sklearn
import nltk
import pickle
import fitz
from sklearn.feature_extraction.text import CountVectorizer
from train import input_process


def load_model_and_vectorizer():
    model = pickle.load(open("classifier.model", "rb"))
    vectorize = pickle.load(open("vectorizer.pickle", "rb"))
    return model, vectorize


if __name__ == "__main__":
    model, vectorize = load_model_and_vectorizer()
    path = input("Enter path of file: ")
    doc = fitz.open(path)
    content = ''
    for page in range(len(doc)):
        content = content + doc[page].get_text()

    content = input_process(content)
    content = vectorize.transform([content])
    pred = model.predict(content)
    if pred[0] == 1:
        print('This document is about AI')
    else:
        print('This document is about WEB')
