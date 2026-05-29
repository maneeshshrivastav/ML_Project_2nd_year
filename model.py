# this will be our model file

# TO-DO:
# K-NEAREST-NEIHBOURS
# SUPPORT VECTOR MACHINE
# DECISION TREE
# NEURAL NETWORK
# NAIVE BAYES

# Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Models
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import LinearSVC as svm
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.neural_network import MLPClassifier as neuron
from sklearn.naive_bayes import MultinomialNB as bayes

from preprocessing import csv_to_dataframe


# Load preprocessed data
x, y = csv_to_dataframe('train_emotion.csv')

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# model dict
models = {
    "kNN": kNN(n_neighbors=5),
    "SVM": svm(random_state=42),
    "Decision Tree": tree(random_state=42),
    "Neural Network": neuron(hidden_layer_sizes=(100,), max_iter=300, random_state=42), # Multi-Layer Perceptron
    "Naive Bayes": bayes()
}


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