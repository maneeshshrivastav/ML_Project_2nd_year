import pandas as pd
import numpy as np
import re # regex
import nltk # Natural Language Toolkit (text processing)
from nltk.corpus import stopwords # common words: "the", "is", (not useful)
from nltk.stem import WordNetLemmatizer # converts words to base form
from sklearn.feature_extraction.text import TfidfVectorizer # converts text into numerical features
from sklearn.preprocessing import LabelEncoder # converts labels into numbers

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
lemmatizer = WordNetLemmatizer() # create lemmatizer?

df = pd.read_csv('train_emotion.csv') # put .csv in a dataframe

# print(df.head())
# print("\nColumns:", df.columns)

texts = df['text'] # input text
labels = df['emotion'] # output labels

def clean_text(text): # cleans input sentences

    text = text.lower() # make all chars lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text) # remove non-alphabetical chars
    words = text.split() # partition sentence into words
    
    # remove stopwords and apply lemmatization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(words) # rejoin words into sentence

# clean and print dataset
texts = texts.apply(clean_text)
print("\nSample of cleaned text:")
print(texts.head())

# create TF-IDF vectorizer and keep top 5000 important words
vectorizer = TfidfVectorizer(max_features=5000)

# learn vocabulary + transform text into numeric matrix
x = vectorizer.fit_transform(texts)
# print("\nTF-IDF shape:", x.shape) # print matrix dimensions

# encode emotion labels into numbers i.e sadness = 0, ...
label_encoder = LabelEncoder() # create encoder
y = label_encoder.fit_transform(labels)

print("\nClasses:", label_encoder.classes_) # print classes
print("\nFinal processed data:")
print("Feature matrix:", x.shape)
print("Labels:", y.shape)