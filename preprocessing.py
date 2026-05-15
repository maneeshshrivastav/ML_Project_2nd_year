import pandas as pd
import numpy as np
import re # regex
import contractions
import nltk # Natural Language Toolkit (text processing)
from nltk.corpus import stopwords # common words: "the", "is", (not useful)
from nltk.stem import WordNetLemmatizer # converts words to base form
from sklearn.feature_extraction.text import TfidfVectorizer # converts text into numerical features
from sklearn.preprocessing import LabelEncoder # converts labels into numbers

def get_nltk_data(): # retrieves stop-words and base forms
    nltk.data.path.append('./nltk_data')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', download_dir='./nltk_data') # download stop-words dictionary
    stop_words = set(stopwords.words('english')) # load stop-words

    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet', download_dir='./nltk_data') # download base-words dictionary

    return stop_words

def csv_to_dataframe(stop_words, csv_file='train_emotion.csv'):
    # -----------------------------
    # TRANSFORM CSV INTO DATAFRAME
    # -----------------------------
    df = pd.read_csv(csv_file) # put .csv in a dataframe

    # print(df.head())
    # print("\nColumns:", df.columns)

    text = df['text'] # input text
    labels = df['emotion'] # output labels

    # -----------------------------
    # CLEAN TEXT COLUMN
    # -----------------------------
    lemmatizer = WordNetLemmatizer() # create lemmatizer?

    # fix contractions, make all lowercase, remove non-alphabetical chars, split
    text = re.sub(r'[^a-zA-Z\s]', '', contractions.fix(text).lower()).split()
    
    # remove stopwords and apply lemmatization
    words = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    text = " ".join(words) # rejoin words into sentence
    # print dataset
    print("\nSample of cleaned text:")
    print(words.head())

    # -----------------------------
    # VECTORIZE DATA
    # -----------------------------
    # create TF-IDF vectorizer and keep top 5000 important words
    vectorizer = TfidfVectorizer(max_features=5000)

    # learn vocabulary + transform text into numeric matrix
    x = vectorizer.fit_transform(text)
    # print("\nTF-IDF shape:", x.shape) # print matrix dimensions

    # encode emotion labels into numbers i.e sadness = 0, ...
    label_encoder = LabelEncoder() # create encoder
    y = label_encoder.fit_transform(labels)

    print("\nClasses:", label_encoder.classes_) # print classes
    print("\nFinal processed data:")
    print("Feature matrix:", x.shape)
    print("Labels:", y.shape)

    return x, y