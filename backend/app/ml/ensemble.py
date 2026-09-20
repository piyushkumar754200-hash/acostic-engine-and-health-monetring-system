import numpy as np
from typing import Dict, List, Any

class EnsembleModel:
    def __init__(self, voting_strategy: str = "weighted"):
        self.model_id = "ensemble"
        self.name = "Weighted Soft Voting Ensemble"
        self.voting_strategy = voting_strategy # "soft", "hard", "weighted"

    def combine_predictions(
        self,
        model_predictions: Dict[str, Dict[str, Any]],
        classes: List[str],
        model_weights: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Combines model predictions into an ensemble output.
        model_predictions format:
        {
          "rf": {"probabilities": [0.8, 0.1, 0.05, 0.05, 0.0], "prediction": "normal", "confidence": 0.8},
          "svm": {"probabilities": [0.9, 0.05, 0.05, 0.0, 0.0], "prediction": "normal", "confidence": 0.9},
          ...
        }
        """
        if not model_predictions:
            return {
                "predicted_class": "normal",
                "confidence": 0.5,
                "model_agreement": 0.0,
                "probabilities": [0.2] * len(classes)
            }

        num_classes = len(classes)
        
        # Default equal weights if not specified
        if model_weights is None:
            model_weights = {m_id: 1.0 for m_id in model_predictions.keys()}

        predicted_indices = []
        prob_accum = np.zeros(num_classes)
        total_weight = 0.0

        for m_id, res in model_predictions.items():
            probs = np.array(res["probabilities"])
            w = model_weights.get(m_id, 1.0)
            
            # Predict index
            pred_idx = int(np.argmax(probs))
            predicted_indices.append(pred_idx)

            if self.voting_strategy in ["soft", "weighted"]:
                prob_accum += probs * w
                total_weight += w

        if total_weight > 0:
            ensemble_probs = prob_accum / total_weight
        else:
            ensemble_probs = np.ones(num_classes) / num_classes

        # Final prediction
        if self.voting_strategy == "hard":
            # Majority vote count
            counts = np.bincount(predicted_indices, minlength=num_classes)
            winning_idx = int(np.argmax(counts))
            confidence = float(counts[winning_idx] / len(predicted_indices))
        else:
            winning_idx = int(np.argmax(ensemble_probs))
            confidence = float(ensemble_probs[winning_idx])

        # Compute Model Agreement Percentage (% of models agreeing with the ensemble winning class)
        agree_count = sum(1 for idx in predicted_indices if idx == winning_idx)
        agreement_percentage = float((agree_count / len(predicted_indices)) * 100.0)

        winning_class = classes[winning_idx] if winning_idx < len(classes) else "normal"

        return {
            "predicted_class": winning_class,
            "predicted_index": winning_idx,
            "confidence": round(confidence, 4),
            "model_agreement_pct": round(agreement_percentage, 1),
            "voting_strategy": self.voting_strategy,
            "class_probabilities": {classes[i]: round(float(ensemble_probs[i]), 4) for i in range(num_classes)}
        }
