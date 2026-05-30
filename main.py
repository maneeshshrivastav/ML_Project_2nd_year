import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer # converts text into numerical features
from sklearn.preprocessing import LabelEncoder # converts labels into numbers
from sklearn.model_selection import train_test_split # Evaluation
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import csv_to_dataframe
from model import models
# TO-DO: TRAIN / TEST, CROSS-VALIDATION, ENSEMBLE

try: # load preprocessed dataset
    df = pd.read_csv("preprocessed.csv")
    # create TF-IDF vectorizer and keep top 5000 important words
    vectorizer = TfidfVectorizer(max_features=5000,
    ngram_range=(1,2))
    label_encoder = LabelEncoder() # create encoder
        
    # learn vocabulary + transform text into numeric matrix
    x = vectorizer.fit_transform(df['text'])
    # encode emotion labels into numbers i.e sadness = 0, ...
    y = label_encoder.fit_transform(df['emotion'])
except FileNotFoundError:
    x, y = csv_to_dataframe()

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# train and evaluate models
for name, model in models.items():

    print(f"\n{'=' * 15} {name} {'=' * 15}")

    # Train model
    model.fit(x_train, y_train)

    # Predict labels
    y_pred = model.predict(x_test)

    # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))