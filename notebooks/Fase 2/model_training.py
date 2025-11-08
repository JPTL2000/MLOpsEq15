from sklearn.base import BaseEstimator

class ModelTrainer(BaseEstimator):
    def __init__(self, model_class, model_params = {}):
        self.model_class = model_class
        self.model_params = model_params
        self.model = None
 
    def fit(self, X, y):
        self.model = self.model_class(**self.model_params)
        self.model.fit(X, y)
        return self
 
    def predict(self, X):
        return self.model.predict(X)
