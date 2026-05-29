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

    try:
        nltk.data.find('corpora/wordnet.zip')
    except LookupError:
        nltk.download('wordnet', download_dir='./nltk_data') # download base-words dictionary

    return set(stopwords.words('english')), WordNetLemmatizer()

def clean_text(text, stop_words, lemmatizer):
    text = contractions.fix(str(text))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.split(" ")

    c_text = []
    for word in text:
        if word not in stop_words: # lemmatize and remove stop words
            c_text.append(lemmatizer.lemmatize(word))

    return " ".join(c_text)

def csv_to_dataframe(csv_file='train_emotion.csv'):
    stop_words, lemmatizer = get_nltk_data()
    df = pd.read_csv(csv_file, names=["text", "emotion"]) # csv to dataframe
    df["text"] = df["text"].apply(lambda x: clean_text(x, stop_words, lemmatizer)) # clean text column
    print(df.head())

    df.to_csv("preprocessed.csv", index=False)

    text = df['text'] # text column handle
    labels = df['emotion'] # label column handle
    

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

if __name__ == "__main__":
    csv_to_dataframe()