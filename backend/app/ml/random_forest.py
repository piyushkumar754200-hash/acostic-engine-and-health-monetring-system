import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from app.config import MODELS_DIR

class RandomForestModel:
    def __init__(self, n_estimators: int = 100, max_depth: int = 15, min_samples_split: int = 2):
        self.model_id = "rf"
        self.name = "Random Forest Classifier"
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        self.is_trained = False
        self.classes_ = []

    def train(self, X: np.ndarray, y: np.ndarray, classes: list):
        self.model.fit(X, y)
        self.is_trained = True
        self.classes_ = classes
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model is not trained.")
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model is not trained.")
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def get_feature_importances(self) -> np.ndarray:
        if self.is_trained and hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return np.array([])

    def save(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "rf_model.joblib")
        joblib.dump({"model": self.model, "classes": self.classes_}, filepath)

    def load(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "rf_model.joblib")
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.model = data["model"]
            self.classes_ = data["classes"]
            self.is_trained = True
            return True
        return False
