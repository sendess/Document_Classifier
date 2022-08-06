import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import string
import nltk
from nltk.corpus import stopwords
import fitz
import pickle


nltk.download('stopwords')
vectorizer = CountVectorizer()


def pre_process_df():
    f_df = pd.DataFrame(columns=['Text', 'Label'])
    df = pd.read_csv('Dataset.csv')
    # f_df['Text'] = df['Text']
    # f_df['Label'] = df['Label']
    df.drop("Unnamed: 0", axis = 1, inplace = True)
    f_df = df
    return f_df



def input_process(text):
    translator = str.maketrans('', '', string.punctuation)
    nopunc = text.translate(translator)
    words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(words)


def remove_stop_words(text):
    # final_input = []
    # for lines in text:
    #     lines = input_process(lines)
    #     final_input.append(lines)
    return [input_process(line) for line in text]



def train_model(df):
    input = remove_stop_words(df['Text'])
    df['Text'] = input
    output = df.Label
    input = vectorizer.fit_transform(input)
    nb = MultinomialNB()
    nb.fit(input, output)
    # print(nb.predict(input[0]))
    # print(nb.predict(input[-1]))
    return nb



if __name__ == '__main__':
    df = pre_process_df()
    model = train_model(df)
    pickle.dump(model, open('classifier.model', 'wb'))
    pickle.dump(vectorizer, open('vectorizer.pickle', 'wb'))