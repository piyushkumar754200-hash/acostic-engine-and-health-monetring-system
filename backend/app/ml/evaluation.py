import time
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Any

class ModelEvaluator:
    @staticmethod
    def evaluate_model(model_obj, X_test, y_test, class_names: List[str]) -> Dict[str, Any]:
        """
        Evaluates a trained model object on X_test, y_test.
        Records inference time and metrics.
        """
        start_time = time.time()
        preds = model_obj.predict(X_test)
        inference_time_ms = ((time.time() - start_time) / len(y_test)) * 1000.0

        acc = float(accuracy_score(y_test, preds))
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average='macro', zero_division=0)
        
        cm = confusion_matrix(y_test, preds, labels=list(range(len(class_names))))

        return {
            "model_id": getattr(model_obj, "model_id", "unknown"),
            "model_name": getattr(model_obj, "name", "Model"),
            "accuracy": round(acc, 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "inference_time_ms": round(inference_time_ms, 2),
            "confusion_matrix": cm.tolist()
        }
