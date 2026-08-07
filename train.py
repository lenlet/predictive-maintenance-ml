"""Train the models once, offline, and write artifacts.joblib for the app to load.

Run this after changing anything in the modelling pipeline:

    python train.py

The app works without the artifact -- it falls back to training on startup -- but
that costs ~43s on every cold start, which on Streamlit's free tier happens again
every time the app wakes from sleep. Precomputing turns that into ~0.04s.

What is stored is deliberately small (~0.1 MB): the deployed model, and the
*results* of the expensive cross-validation rather than the four fitted models.
The comparison tab only ever needed the probability arrays and the metrics.
"""

import time
import warnings

import joblib
import numpy as np
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC

from app_core import ARTIFACT_PATH, FEATURES, prepare_data

warnings.filterwarnings("ignore")


def main():
    t0 = time.time()
    print("loading data and engineering features...")
    _, _, _, _, _, y_train, y_test, X_train_s, X_test_s, _ = prepare_data()

    candidates = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=42),
        "SVM (RBF)":           SVC(kernel="rbf", probability=True, class_weight="balanced",
                                   C=10, gamma="scale", random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results, models = {}, {}

    for name, clf in candidates.items():
        t = time.time()
        clf.fit(X_train_s, y_train)
        y_pred = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)[:, 1]
        cv_roc = cross_val_score(clf, X_train_s, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        models[name] = clf
        # Note: the fitted estimator is intentionally NOT stored here. Only the
        # deployed model is kept; the rest survive as their metrics and outputs.
        results[name] = {
            "roc_auc":     roc_auc_score(y_test, y_proba),
            "avg_prec":    average_precision_score(y_test, y_proba),
            "brier":       brier_score_loss(y_test, y_proba),
            "cv_roc_mean": cv_roc.mean(),
            "cv_roc_std":  cv_roc.std(),
            "y_pred":      y_pred,
            "y_proba":     y_proba,
        }
        print(f"  {name:22s} roc_auc {results[name]['roc_auc']:.4f}   {time.time()-t:5.1f}s")

    # The deployed model is whichever candidate won on Average Precision -- the
    # metric that matters at a 3.4% positive rate. Chosen by measurement, so that
    # swapping in a better model later needs no edit here.
    best = max(results, key=lambda n: results[n]["avg_prec"])
    final_model = models[best]
    print(f"\nbest by Average Precision: {best} ({results[best]['avg_prec']:.4f})")

    print("computing SHAP values...")
    t = time.time()
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_s)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values
    print(f"  shap {time.time()-t:.1f}s  shape {np.shape(shap_vals)}")

    joblib.dump(
        {
            "final_model":  final_model,
            "best_name":    best,
            "results":      results,
            "shap_vals":    shap_vals,
            "features":     FEATURES,
            "sklearn_hint": "regenerate with `python train.py` if sklearn warns about version mismatch",
        },
        ARTIFACT_PATH,
        compress=3,
    )
    size = ARTIFACT_PATH.stat().st_size / 1e6
    print(f"\nwrote {ARTIFACT_PATH.name} ({size:.2f} MB) in {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
