import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Limpia y prepara los datos para su uso en un pipeline de Machine Learning.

    Este transformador:
      - Convierte columnas object a numéricas (cuando sea posible)
      - Maneja valores faltantes
      - Elimina columnas irrelevantes o problemáticas (por ejemplo 'url', 'mixed_type_col')
    """

    def __init__(self, drop_columns=None, fill_strategy='median'):
        """
        Parámetros:
        -----------
        drop_columns : list, opcional
            Columnas a eliminar (por defecto ['url', 'mixed_type_col'])
        fill_strategy : str, opcional
            Estrategia para rellenar valores faltantes ('median', 'mean', 'mode')
        """
        self.drop_columns = drop_columns if drop_columns is not None else ['url', 'mixed_type_col']
        self.fill_strategy = fill_strategy
        self.numeric_columns_ = None

    def fit(self, X, y=None):
        """Ajusta el transformador a los datos (por compatibilidad con Pipeline)."""
        X = X.copy()

        # Guardamos columnas numéricas para usar en transform()
        self.numeric_columns_ = X.select_dtypes(include=np.number).columns.tolist()

        # Calculamos valores de reemplazo para NaN según la estrategia
        if self.fill_strategy == 'median':
            self.fill_values_ = X[self.numeric_columns_].median()
        elif self.fill_strategy == 'mean':
            self.fill_values_ = X[self.numeric_columns_].mean()
        elif self.fill_strategy == 'mode':
            self.fill_values_ = X[self.numeric_columns_].mode().iloc[0]
        else:
            raise ValueError("fill_strategy debe ser 'median', 'mean' o 'mode'")

        return self

    def transform(self, X):
        """Realiza la limpieza de los datos."""
        X = X.copy()

        # 1. Eliminar columnas irrelevantes
        X = X.drop(columns=[col for col in self.drop_columns if col in X.columns], errors='ignore')

        # 2. Intentar convertir columnas object a numéricas (cuando sea posible)
        for col in X.select_dtypes(include='object').columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

        # 3. Rellenar valores faltantes
        if self.numeric_columns_ is not None:
            X[self.numeric_columns_] = X[self.numeric_columns_].fillna(self.fill_values_)

        # 4. Reemplazar infinitos por NaN y volver a rellenar
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(self.fill_values_, inplace=True)

        return X
