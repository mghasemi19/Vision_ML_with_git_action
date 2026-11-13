"""Candidate classifier, reference classifier, and evaluation metrics."""

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vision_mlops.data import to_features

# This reference is a naive baseline, NOT an incumbent production model.
MIN_ACCURACY = 0.94
MIN_MACRO_F1 = 0.94
MIN_CLASS_RECALL = 0.80
MIN_GAIN_OVER_REFERENCE_F1 = 0.20


def new_candidate():
    # Fit scaler exclusively on train: prevents train/validation/test leakage.
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])


def fit_models(images, labels, train_idx):
    X_train = to_features(images[train_idx])
    y_train = labels[train_idx]
    candidate = new_candidate()
    reference = DummyClassifier(strategy="most_frequent")
    candidate.fit(X_train, y_train)
    reference.fit(X_train, y_train)
    return candidate, reference


def metrics_for(model, images, labels):
    y_pred = model.predict(to_features(images))
    per_class = recall_score(labels, y_pred, average=None, labels=list(range(10)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(labels, y_pred)),
        "macro_f1": float(f1_score(labels, y_pred, average="macro", zero_division=0)),
        "per_class_recall": [float(v) for v in per_class],
        "confusion_matrix": confusion_matrix(labels, y_pred, labels=list(range(10))).tolist(),
        "num_samples": int(len(labels)),
    }


def quality_failures(candidate_metrics, reference_metrics):
    problems = []
    if candidate_metrics["accuracy"] < MIN_ACCURACY:
        problems.append(f"accuracy {candidate_metrics['accuracy']:.3f} < {MIN_ACCURACY}")
    if candidate_metrics["macro_f1"] < MIN_MACRO_F1:
        problems.append(f"macro F1 {candidate_metrics['macro_f1']:.3f} < {MIN_MACRO_F1}")
    worst = min(candidate_metrics["per_class_recall"])
    if worst < MIN_CLASS_RECALL:
        problems.append(f"worst class recall {worst:.3f} < {MIN_CLASS_RECALL}")
    gain = candidate_metrics["macro_f1"] - reference_metrics["macro_f1"]
    if gain < MIN_GAIN_OVER_REFERENCE_F1:
        problems.append(f"F1 improvement over naive reference {gain:.3f} < {MIN_GAIN_OVER_REFERENCE_F1}")
    return problems
