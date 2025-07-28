import numpy as np


import math
def proportional_split(
        num_to_split: int,
        proportions: np.ndarray
) -> list:
    """
    Splits a number to proportional subsets.
    
    :param num_to_split: Number to be proportionally distributed.
    :param proportions: Array of proportions.
    :return: Array of proportional splits.
    """

    raw = [p*num_to_split for p in proportions]
    results = [math.floor(x) for x in raw]
    remainder = num_to_split - sum(results)

    fracs = [r - f for (r, f) in zip(raw, results)]
    idxs = sorted(
        range(len(proportions)), key=lambda i: fracs[i], reverse=True)

    for idx in idxs[:remainder]:
        results[idx] += 1

    return results


from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix,
                             balanced_accuracy_score
                             )
from typing import Any
def evaluate_model(
        estimator: Any,
        X_test: np.ndarray,
        Y_test: np.ndarray
) -> dict[str, float]:
    """
    Evaluates a trained estimator, with basic sklearn metrics.

    :param estimator: Trained estimator to be evaluated.
    :param X_test: Test features array to predict new labels.
    :param Y_test: Real labels to compare with trained ones.
    :return: Dictionary of evaluation results.
    """

    Y_pred = estimator.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
    return {
        'accuracy': accuracy_score(Y_test, Y_pred),
        'precision': precision_score(Y_test, Y_pred, average='binary', pos_label=1),
        'recall': recall_score(Y_test, Y_pred, average='binary', pos_label=1),
        'f1_score': f1_score(Y_test, Y_pred, average='binary', pos_label=1),
        'specificity': tn / (tn + fp),
        'balanced_accuracy': balanced_accuracy_score(Y_test, Y_pred)
    }