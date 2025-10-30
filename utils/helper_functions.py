import numpy as np
import pandas as pd

import math
def proportional_split(
        num_to_split: int,
        proportions: np.ndarray
) -> list:
    """
    Splits an integer into proportional subsets based on an array-like of proportions.

    This function multiplies each proportion by `num_to_split`, takes the floor of each value,
    and then distributes any remaining units to the elements with the largest fractional parts.

    Args:
        num_to_split (int): Total integer value to be proportionally distributed.
        proportions (array-like): One-dimensional array-like object
            (e.g., list, tuple, np.ndarray) of non-negative proportions.

    Returns:
        list[int]: A list of integer counts representing the proportional splits.
            The sum of all values equals `num_to_split`.

    Notes:
        - The final adjustment (adding 1 to largest fractions) is a rounding correction
          to ensure the sum of resulting integers matches `num_to_split`.
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

from scipy.stats import spearmanr
def spearman_with_bootstrap(
        list_1: list,
        list_2: list,
        n_boot: int = 1000,
        random_seed: int = 0
) -> dict:
    """
    Computes the Spearman rank correlation between two array-like inputs and estimates
    a 95% bootstrap confidence interval.

    Args:
        list_1 (array-like): First sequence of numeric values (e.g., list, tuple, np.ndarray, pd.Series).
            Must be the same length as `list_2`.
        list_2 (array-like): Second sequence of numeric values (e.g., list, tuple, np.ndarray, pd.Series).
            Must be the same length as `list_1`.
        n_boot (int, optional): Number of bootstrap resamples used to estimate the confidence interval.
            Default is 1000.
        random_seed (int, optional): Seed for the random number generator to ensure reproducibility.
            Default is 0.

    Returns:
        dict: Dictionary with the following keys:
            - 'rho' (float): Spearman correlation coefficient between the two inputs.
            - 'pval' (float): P-value testing the null hypothesis of no correlation.
            - 'n' (int): Number of paired observations used.
            - 'ci' (tuple[float, float]): 95% bootstrap confidence interval (lower, upper) for the correlation.
    """

    N = len(list_1)

    rho, pval = spearmanr(list_1, list_2)

    rng = np.random.RandomState(random_seed)
    boots = []

    array_1 = np.array(list_1)
    array_2 = np.array(list_2)

    for i in range(n_boot):
        idx = rng.randint(0, N, N)
        r, _ = spearmanr(array_1[idx], array_2[idx])
        boots.append(r)

    low, high = np.percentile(boots, [2.5, 97.5])
    return {'rho': float(rho), 'pval': float(pval), 'n': N, 'ci': (float(low), float(high))}

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
    Evaluates a trained estimator using standard scikit-learn metrics.

    Args:
        estimator: A trained scikit-learn-compatible estimator with a `predict` method.
        X_test (array-like): Feature matrix for the test set (e.g., np.ndarray, pd.DataFrame, list of lists).
            Must match the input format used during training.
        Y_test (array-like): True labels for the test set (e.g., np.ndarray, pd.Series, list).
            Must have the same length as `X_test`.

    Returns:
        dict: Dictionary of evaluation metrics such as accuracy, precision, recall, F1-score,
            and any additional computed metrics depending on the implementation.
    """

    if X_test.shape[0] != Y_test.shape[0]:
        raise ValueError(f"X_test and Y_test must have same number of samples: ({X_test.shape[0]} != {Y_test.shape[0]}).")

    if not hasattr(estimator, "predict"):
        raise AttributeError("Estimator must implement a 'predict' method.")

    Y_pred = estimator.predict(X_test)

    labels = np.unique(Y_test)
    if labels.shape[0] == 2:
        tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()

        return {
            'accuracy': accuracy_score(Y_test, Y_pred),
            'precision': precision_score(Y_test, Y_pred, average='binary', pos_label=1, zero_division=0),
            'recall': recall_score(Y_test, Y_pred, average='binary', pos_label=1, zero_division=0),
            'f1_score': f1_score(Y_test, Y_pred, average='binary', pos_label=1, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) != 0 else float('nan'),
            'balanced_accuracy': balanced_accuracy_score(Y_test, Y_pred)
        }
    else:
        return {
            'accuracy': accuracy_score(Y_test, Y_pred),
            'precision': precision_score(Y_test, Y_pred, average='macro', zero_division=0),
            'recall': recall_score(Y_test, Y_pred, average='macro', zero_division=0),
            'f1_score': f1_score(Y_test, Y_pred, average='macro', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(Y_test, Y_pred)
        }

import matplotlib.pyplot as plt
def plot_metrics(
        df: pd.DataFrame,
        selected: list[str],
        x_axis: str,
        x_step: float,
        x_label: str = None,
        y_label: str = None,
        y_range: tuple[float, float] = None,
        legend_loc: tuple[float, float] = (0.04, 0.06),
        fig_size: tuple[int, int] = (10, 6),
        title: str = None
) -> None:
    """
    Plots selected performance metrics from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing performance metrics. Must contain the columns specified in `selected` and `x_axis`.
        selected (list[str]): List of column names from `df` to be plotted.
        x_axis (str): Name of the DataFrame column to use for the x-axis.
        x_step (int | float): Interval between x-axis ticks.
        x_label (str): Label to display on the x-axis.
        y_label (str, optional): Label to display on the y-axis.
        y_range (tuple[float, float], optional): Tuple (y_min, y_max) defining y-axis limits.
        legend_loc (tuple[float, float], optional): Location of the legend as (x, y) in axes coordinates.
            Deafult is (0.04, 0.06).
        fig_size (tuple[float, float], optional): Figure size as (width, height) in inches.
            Default is (10, 6).
        title (str, optional): Title for the plot.

    Returns:
        None
    """

    if df.empty:
        raise ValueError('Metrics DataFrame is empty.')

    if x_axis not in df.columns:
        raise ValueError(f'Feature {x_axis} does not exist.')

    for metric in selected:
        if metric not in df.columns:
            raise ValueError(f'Metric {metric} does not exist.')

    plt.style.use('default')
    plt.figure(figsize=fig_size, dpi=500)

    for metric in selected:
        plt.plot(df[x_axis], df[metric], label=metric)

    if y_range is None:
        metric_min = round(df[selected].min().min(), 2)
        metric_max = round(df[selected].max().max(), 2)
    else:
        metric_min = y_range[0] if y_range[0] is not None else round(df[selected].min().min(), 2)
        metric_max = y_range[1] if y_range[1] is not None else round(df[selected].max().max(), 2)

    plt.xticks(np.arange(df[x_axis].min(), df[x_axis].max() + x_step, x_step))
    plt.xticks(rotation=60, size=8)
    plt.yticks(np.arange(metric_min, metric_max + 0.01, 0.01))
    plt.yticks(rotation=30, size=8)

    if x_label is not None:
        plt.xlabel(x_label, size=10)
    plt.xlim(df[x_axis].min(), df[x_axis].max())

    if y_label is not None:
        plt.ylabel(y_label, size=10)
    plt.ylim(metric_min - 0.005, metric_max + 0.005)

    plt.legend(loc='lower left', prop={'size': 12}, bbox_to_anchor=legend_loc)

    if title is not None:
        plt.title(title)

    plt.grid(True)
    plt.tight_layout()

import seaborn as sns
def plot_heatmap(
        df: pd.DataFrame,
        x_axis: str,
        y_axis: str,
        heat: str,
        x_label: str = None,
        y_label: str = None,
        c_map: str = None,
        v_range: tuple[float, float] = None,
        annotate: bool = False,
        fmt: str = '.2f',
        fig_size: tuple[int, int] = (10, 6),
        title: str = None
) -> None:
    """
    Creates and displays a heatmap from a DataFrame pivot.

    Args:
        df (pd.DataFrame): Source DataFrame containing performance metrics. Must contain the columns specified in `x_axis`, `y_axis` and `heat`.
        x_axis (str): Column name to use as columns in the pivot (x-axis).
        y_axis (str): Column name to use as index in the pivot (y-axis).
        heat (str): Column name whose values populate the heatmap cells.
        x_label (str, optional): Label for the x-axis.
        y_label (str, optional): Label for the y-axis.
        c_map (str, optional): Colormap name to use for the heatmap.
            Default same as in a Seaborn library.
        v_range (tuple[float, float], optional): Tuple (vmin, vmax) defining color scale limits.
            If None, limits are inferred from the range of `df[heat]`.
        annotate (bool, optional): Whether to annotate each cell with its numeric value.
            Default is False.
        fmt (str, optional): String format for annotations. Default is '.2f'.
        fig_size (tuple[int, int], optional): Figure size as (width, height) in inches.
            Default is (10, 6).
        title (str, optional): Title for the plot.

    Returns:
        None
    """

    if df.empty:
        raise ValueError("DataFrame is empty.")
    for column in (x_axis, y_axis, heat):
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist.")

    dfc = df.copy()
    pivot = dfc.pivot_table(index=y_axis, columns=x_axis, values=heat, aggfunc='mean')

    try:
        pivot = pivot.sort_index(ascending=False)
    except TypeError:
        pass
    try:
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    except TypeError:
        pass

    if v_range is None:
        metric_min = round(df[heat].min(), 2)
        metric_max = round(df[heat].max(), 2)
    else:
        metric_min = v_range[0] if v_range[0] is not None else round(df[heat].min(), 2)
        metric_max = v_range[1] if v_range[1] is not None else round(df[heat].max(), 2)

    plt.figure(figsize=fig_size, dpi=500)
    sns.heatmap(
        pivot,
        cmap=c_map,
        vmin=metric_min,
        vmax=metric_max,
        center=(metric_min + metric_max)*0.50,
        annot=annotate,
        fmt=fmt,
        cbar_kws={'label': heat},
        linewidths=0.25,
        linecolor='white',
        square=False
    )

    plt.xticks(rotation=60)
    plt.yticks(rotation=30)

    if x_label is not None:
        plt.xlabel(x_label, size=10)

    if y_label is not None:
        plt.ylabel(y_label, size=10)

    if title is not None:
        plt.title(title)

    plt.tight_layout()