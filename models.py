from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neural_network import MLPClassifier

class SVMClassifier:
    def get_best_model(self, X, y):
        return SVC(probability=True, kernel='rbf', C=1, gamma='scale')

class RandomForestClassifier:
    def get_best_model(self, X, y):
        return RF(n_estimators=100, max_depth=3, random_state=42)

class NeuralNetworkClassifier:
    def get_best_model(self, X, y):
        return MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
