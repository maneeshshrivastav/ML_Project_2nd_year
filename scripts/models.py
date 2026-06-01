# models to be evaluated: kNN, SVM, D-Tree, Neural-Network, Naive-Bayes

from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import LinearSVC as svm
from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.neural_network import MLPClassifier as neuron
from sklearn.naive_bayes import MultinomialNB as bayes

# model dict
models = {
    "kNN": kNN(n_neighbors=5),
    "Decision Tree": tree(random_state=42),
    "Neural Network": neuron(hidden_layer_sizes=(32,), max_iter=50, early_stopping=True, random_state=42), # Multi-Layer Perceptron
    "Naive Bayes": bayes(),
    "SVM": svm(random_state=42)
}
