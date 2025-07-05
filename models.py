from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neural_network import MLPClassifier

class SVMClassifier:
    def __init__(self):
        self.model = SVC()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class RandomForestClassifier:
    def __init__(self):
        self.model = RF()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class NeuralNetworkClassifier:
    def __init__(self):
        self.model = MLPClassifier(max_iter=1000)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
