import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from app.config import MODELS_DIR

class SVMModel:
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', gamma: str = 'scale'):
        self.model_id = "svm"
        self.name = "Support Vector Machine (SVM)"
        self.scaler = StandardScaler()
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=42
        )
        self.is_trained = False
        self.classes_ = []

    def train(self, X: np.ndarray, y: np.ndarray, classes: list):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        self.classes_ = classes
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model is not trained.")
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model is not trained.")
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def save(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "svm_model.joblib")
        joblib.dump({"model": self.model, "scaler": self.scaler, "classes": self.classes_}, filepath)

    def load(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, "svm_model.joblib")
        if os.path.exists(filepath):
            data = joblib.load(filepath)
            self.model = data["model"]
            self.scaler = data["scaler"]
            self.classes_ = data["classes"]
            self.is_trained = True
            return True
        return False
