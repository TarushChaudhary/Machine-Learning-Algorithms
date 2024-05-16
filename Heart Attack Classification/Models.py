import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

class LogisticRegressionCoded:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        
        for _ in range(self.num_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls
        class KNearestNeighbors:
            def __init__(self, k=5):
                self.k = k
                self.X_train = None
                self.y_train = None
            
            def fit(self, X_train, y_train):
                self.X_train = X_train
                self.y_train = y_train
            
            def predict(self, X_test):
                distances = pairwise_distances(X_test, self.X_train)
                sorted_indices = np.argsort(distances, axis=1)
                k_nearest_indices = sorted_indices[:, :self.k]
                k_nearest_labels = self.y_train[k_nearest_indices]
                y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=k_nearest_labels)
                return y_pred
    


