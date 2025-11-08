from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd

class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, handle_unknown="ignore", scale_numeric=True):
        self.handle_unknown = handle_unknown
        self.scale_numeric = scale_numeric
        self.preprocessor = None

    def fit(self, X, y=None):
        X = X.copy()
        num = X.select_dtypes(include=["number"]).columns.tolist()
        cat = X.select_dtypes(exclude=["number"]).columns.tolist()

        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False) if self.scale_numeric else "passthrough")
        ])
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown=self.handle_unknown))
        ])
        self.preprocessor = ColumnTransformer([
            ("num", num_pipe, num),
            ("cat", cat_pipe, cat),
        ])
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

