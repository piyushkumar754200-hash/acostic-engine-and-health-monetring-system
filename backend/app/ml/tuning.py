from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from typing import Dict, Any

class HyperparameterTuner:
    @staticmethod
    def tune_random_forest(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X, y)
        return {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_)
        }

    @staticmethod
    def tune_svm(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
        svm = SVC(random_state=42)
        grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X, y)
        return {
            "best_params": grid_search.best_params_,
            "best_score": float(grid_search.best_score_)
        }
