import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


from app.config import DATASET_DIR, MODELS_DIR, FAULT_CLASS_NAMES
from app.audio.processor import AudioProcessor
from app.ml.random_forest import RandomForestModel
from app.ml.svm import SVMModel
from app.ml.cnn_1d import CNN1DModel
from app.ml.cnn_2d import CNN2DModel
from app.ml.lstm import LSTMModel
from app.ml.ensemble import EnsembleModel
from app.ml.evaluation import ModelEvaluator
from app.ml.tuning import HyperparameterTuner

def train_pipeline(dataset_dir: Path = DATASET_DIR, run_tuning: bool = False):
    print("==================================================")
    print(" ENGINE-SENSE AI — TRAINING PIPELINE ")
    print("==================================================")

    processor = AudioProcessor()
    
    # 1. Discover classes & audio files
    if not dataset_dir.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist.")
        return None

    classes = [d.name for d in dataset_dir.iterdir() if d.is_dir()]
    if not classes:
        print(f"No class folders found in {dataset_dir}")
        return None
        
    classes.sort()
    print(f"Discovered {len(classes)} fault classes: {classes}")

    X_tab = []
    X_mel = []
    X_seq = []
    y = []

    file_count = 0
    for class_idx, class_name in enumerate(classes):
        class_folder = dataset_dir / class_name
        audio_files = list(class_folder.glob("*.wav")) + list(class_folder.glob("*.mp3")) + list(class_folder.glob("*.flac"))
        print(f"Loading {len(audio_files)} files for class '{class_name}'...")
        
        for audio_file in audio_files:
            try:
                y_audio, sr = processor.load_audio(str(audio_file))
                feat = processor.extract_features(y_audio, sr)
                
                X_tab.append(feat["tabular_vector"])
                X_mel.append(feat["mel_spectrogram_2d"])
                X_seq.append(feat["sequence_features"])
                y.append(class_idx)
                file_count += 1
            except Exception as e:
                print(f"Failed to process {audio_file}: {e}")

    print(f"\nSuccessfully loaded and extracted acoustic features for {file_count} samples.")

    X_tab = np.array(X_tab)
    X_mel = np.array(X_mel)
    X_seq = np.array(X_seq)
    y = np.array(y)

    # 2. Train/Validation/Test Split (Stratified)
    indices = np.arange(len(y))
    idx_train, idx_test, y_train, y_test = train_test_split(
        indices, y, test_size=0.3, random_state=42, stratify=y
    )

    X_tab_train, X_tab_test = X_tab[idx_train], X_tab[idx_test]
    X_mel_train, X_mel_test = X_mel[idx_train], X_mel[idx_test]
    X_seq_train, X_seq_test = X_seq[idx_train], X_seq[idx_test]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    evaluation_results = []

    # 3. Random Forest
    print("\n[1/5] Training Random Forest...")
    rf_model = RandomForestModel()
    if run_tuning and len(y_train) >= 6:
        print("Running Random Forest Hyperparameter Tuning...")
        best = HyperparameterTuner.tune_random_forest(X_tab_train, y_train)
        print(f"Best RF params: {best['best_params']}")
        rf_model = RandomForestModel(**best['best_params'])
    rf_model.train(X_tab_train, y_train, classes)
    rf_model.save()
    rf_metrics = ModelEvaluator.evaluate_model(rf_model, X_tab_test, y_test, classes)
    evaluation_results.append(rf_metrics)
    print(f"RF -> Accuracy: {rf_metrics['accuracy'] * 100:.1f}%, F1: {rf_metrics['f1_score']:.3f}")

    # 4. Support Vector Machine (SVM)
    print("\n[2/5] Training Support Vector Machine (SVM)...")
    svm_model = SVMModel()
    if run_tuning and len(y_train) >= 6:
        print("Running SVM Hyperparameter Tuning...")
        best = HyperparameterTuner.tune_svm(X_tab_train, y_train)
        print(f"Best SVM params: {best['best_params']}")
        svm_model = SVMModel(**best['best_params'])
    svm_model.train(X_tab_train, y_train, classes)
    svm_model.save()
    svm_metrics = ModelEvaluator.evaluate_model(svm_model, X_tab_test, y_test, classes)
    evaluation_results.append(svm_metrics)
    print(f"SVM -> Accuracy: {svm_metrics['accuracy'] * 100:.1f}%, F1: {svm_metrics['f1_score']:.3f}")

    # 5. 1D CNN
    print("\n[3/5] Training 1D CNN...")
    cnn1d_model = CNN1DModel(epochs=15)
    cnn1d_model.train(X_tab_train, y_train, classes)
    cnn1d_model.save()
    cnn1d_metrics = ModelEvaluator.evaluate_model(cnn1d_model, X_tab_test, y_test, classes)
    evaluation_results.append(cnn1d_metrics)
    print(f"1D CNN -> Accuracy: {cnn1d_metrics['accuracy'] * 100:.1f}%, F1: {cnn1d_metrics['f1_score']:.3f}")

    # 6. 2D CNN
    print("\n[4/5] Training 2D Mel-Spectrogram CNN...")
    cnn2d_model = CNN2DModel(epochs=12)
    cnn2d_model.train(X_mel_train, y_train, classes)
    cnn2d_model.save()
    cnn2d_metrics = ModelEvaluator.evaluate_model(cnn2d_model, X_mel_test, y_test, classes)
    evaluation_results.append(cnn2d_metrics)
    print(f"2D CNN -> Accuracy: {cnn2d_metrics['accuracy'] * 100:.1f}%, F1: {cnn2d_metrics['f1_score']:.3f}")

    # 7. LSTM
    print("\n[5/5] Training LSTM Sequential Model...")
    lstm_model = LSTMModel(epochs=15)
    lstm_model.train(X_seq_train, y_train, classes)
    lstm_model.save()
    lstm_metrics = ModelEvaluator.evaluate_model(lstm_model, X_seq_test, y_test, classes)
    evaluation_results.append(lstm_metrics)
    print(f"LSTM -> Accuracy: {lstm_metrics['accuracy'] * 100:.1f}%, F1: {lstm_metrics['f1_score']:.3f}")

    # 8. Ensemble Evaluation
    print("\nComputing Ensemble Metrics...")
    ensemble = EnsembleModel(voting_strategy="weighted")
    ensemble_preds = []
    ens_start = time.time()

    for i in range(len(y_test)):
        sample_tab = X_tab_test[i]
        sample_mel = X_mel_test[i]
        sample_seq = X_seq_test[i]

        preds_map = {
            "rf": {"probabilities": rf_model.predict_proba(sample_tab)[0].tolist()},
            "svm": {"probabilities": svm_model.predict_proba(sample_tab)[0].tolist()},
            "cnn_1d": {"probabilities": cnn1d_model.predict_proba(sample_tab)[0].tolist()},
            "cnn_2d": {"probabilities": cnn2d_model.predict_proba(sample_mel)[0].tolist()},
            "lstm": {"probabilities": lstm_model.predict_proba(sample_seq)[0].tolist()}
        }

        ens_res = ensemble.combine_predictions(preds_map, classes)
        ensemble_preds.append(ens_res["predicted_index"])

    ens_inference_ms = ((time.time() - ens_start) / len(y_test)) * 1000.0
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    ens_acc = float(accuracy_score(y_test, ensemble_preds))
    ens_prec, ens_rec, ens_f1, _ = precision_recall_fscore_support(y_test, ensemble_preds, average='macro', zero_division=0)
    ens_cm = confusion_matrix(y_test, ensemble_preds, labels=list(range(len(classes))))

    ensemble_metrics = {
        "model_id": "ensemble",
        "model_name": "Weighted Soft Voting Ensemble",
        "accuracy": round(ens_acc, 4),
        "precision": round(float(ens_prec), 4),
        "recall": round(float(ens_rec), 4),
        "f1_score": round(float(ens_f1), 4),
        "inference_time_ms": round(ens_inference_ms, 2),
        "confusion_matrix": ens_cm.tolist()
    }
    evaluation_results.append(ensemble_metrics)
    print(f"Ensemble -> Accuracy: {ensemble_metrics['accuracy'] * 100:.1f}%, F1: {ensemble_metrics['f1_score']:.3f}")

    # 9. Save all metrics to JSON
    summary_data = {
        "is_trained": True,
        "dataset_summary": {
            "total_samples": file_count,
            "train_samples": len(idx_train),
            "test_samples": len(idx_test),
            "classes": classes
        },
        "models": evaluation_results,
        "overall_confusion_matrix": ensemble_metrics["confusion_matrix"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    metrics_file = MODELS_DIR / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nSaved evaluation metrics and model artifacts to {metrics_file}")
    return summary_data

if __name__ == "__main__":
    train_pipeline()
