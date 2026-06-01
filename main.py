import pandas as pd
import re
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer # converts text into numerical features
from sklearn.preprocessing import LabelEncoder # converts labels into numbers
from sklearn.model_selection import train_test_split # Evaluation
from sklearn.metrics import accuracy_score, classification_report
from scripts.preprocessing import csv_to_dataframe, get_nltk_data, clean_text
from scripts.models import models
from sklearn.model_selection import cross_val_score # cross-validation
from sklearn.metrics import confusion_matrix # confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay # confusion matrix visualization
import matplotlib.pyplot as plt # plotting

def train_test(train_csv="train_emotion.csv", test_csv="test.csv"):
    # simple text cleaner without NLTK downloads
    def simple_clean(text):
        text = str(text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = " ".join(text.split())
        return text

    # load train and test files
    train_df = pd.read_csv(train_csv, names=["text", "emotion"])
    
    try:
        test_df = pd.read_csv(test_csv, names=["text"])
    except FileNotFoundError:
        print("\n\n\n==========================================================================================")
        print("FileNotFoundError: Please, place your 'test.csv' file inside the project's root directory.")
        print("==========================================================================================\n")
        exit()

    # clean text
    train_df["text"] = train_df["text"].apply(simple_clean)
    test_df["text"] = test_df["text"].apply(simple_clean)

    # TF-IDF with unigrams + bigrams
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
    label_encoder = LabelEncoder()

    x_train = vectorizer.fit_transform(train_df["text"])
    y_train = label_encoder.fit_transform(train_df["emotion"])
    x_test = vectorizer.transform(test_df["text"])

    # final selected model: SVM
    model = models["SVM"]
    model.fit(x_train, y_train)

    # predict labels
    predictions = model.predict(x_test)
    predicted_labels = label_encoder.inverse_transform(predictions)

    # write predictions
    with open("test_files/predictions.txt", "w") as file:
        for label in predicted_labels:
            file.write(str(label) + "\n")

# ------------------------------------------------------------

    try: # load preprocessed dataset
        df = pd.read_csv("test_files/preprocessed.csv")
    except FileNotFoundError:
        print("Preprocessed data not found.")
        print("Preprocessing data now.")
        csv_to_dataframe()
        df = pd.read_csv("test_files/preprocessed.csv")
    # create TF-IDF vectorizer and keep top 5000 important words
    vectorizer = TfidfVectorizer(max_features=5000,
    ngram_range=(1,2))
    label_encoder = LabelEncoder() # create encoder
        
    # learn vocabulary + transform text into numeric matrix
    x = vectorizer.fit_transform(df['text'])
    # encode emotion labels into numbers i.e sadness = 0, ...
    y = label_encoder.fit_transform(df['emotion'])

    # Split data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    # write scores
    with open("test_files/scores.txt", "w") as file:

        for name, model in models.items():

            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)

            file.write(f"\n{'=' * 15} {name} {'=' * 15}\n")
            file.write(f"Accuracy: {accuracy:.4f}\n")
            file.write("Classification Report:\n")
            file.write(classification_report(y_test, y_pred))
            
            # run additional evaluation for the best model (SVM)
            if name == "SVM":

                # evaluate SVM using 5-fold cross-validation
                scores = cross_val_score(
                    model,
                    x,
                    y,
                    cv=5,
                    scoring='accuracy'
                )

                # print fold accuracies and average accuracy
                print("\n===== SVM 5-Fold Cross Validation =====")
                print("Fold Scores:", scores)
                print("Mean Accuracy:", scores.mean())
                print("Standard Deviation:", scores.std())

                # create confusion matrix for SVM predictions
                cm = confusion_matrix(y_test, y_pred)

                # display confusion matrix
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm,
                    display_labels=label_encoder.classes_
                )

                # plot confusion matrix
                disp.plot(cmap="Blues")
                plt.title("SVM Confusion Matrix")
                plt.show()

if __name__ == "__main__":
    train_test()




