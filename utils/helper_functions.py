import numpy as np
import pandas as pd

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

from scipy.stats import spearmanr
def spearman_with_bootstrap(
        list_1: list,
        list_2: list,
        n_boot: int = 1000,
        random_seed: int =0
) -> dict:
    """
    Computes the Spearman correlation coefficient between two lists
    in the provided DataFrame and estimates a 95% confidence interval using bootstrap resampling.
    :param list_1: The first list to be compared.
    :param list_2: The second list to be compared.
    :param n_boot: Int, optional (default=1000)
                    Number of bootstrap resamples to estimate the confidence interval.
    :param random_seed: Int, optional (default=0)
                    Seed for the random number generator to ensure reproducibility.
    :return: dict with the following keys:
             - 'rho': Spearman correlation coefficient between two lists.
             - 'pval': P-value testing the null hypothesis of no correlation.
             - 'n': Number of parameter pairs used in the calculation.
             - 'ci': tuple (lo, hi), 95% bootstrap confidence interval for the Spearman correlation.
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

import matplotlib.pyplot as plt
def plot_metrics(
        df: pd.DataFrame,
        selected: list[str],
        x_axis: str,
        x_step: float,
        x_label: str = None,
        y_label: str = None,
        y_range: tuple[float, float] = None,
        legend_loc: tuple[float, float] = None,
        fig_size: tuple[int, int] = (10, 6),
        title: str = None
) -> None:
    """
    :param df: DataFrame containing performance metrics.
    :param selected: Performance metrics selected to be plotted.
    :param x_axis: Name of the DataFrame column to plot on the x-axis.
    :param x_step: Interval between ticks.
    :param x_label: Name of the y-axis label to display.
    :param y_label: (optinal) Name of the y-axis label to display.
    :param y_range: (optional) Tuple containing the minimum and maximum values for the y-axis (y_min, y_max).
    :param legend_loc: (optional) Location for the legend, as a pair of (x, y) floats.
    :param fig_size: (optional) Width, height in inches. Default is (10, 6).
    :param title: (optional) Title for the plot.
    :return:
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

    if legend_loc is None:
        legend_loc = (0.04, 0.06)
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
    :param df:
    :param x_axis:
    :param y_axis:
    :param heat:
    :param x_label:
    :param y_label:
    :param c_map:
    :param v_range:
    :param annotate:
    :param fmt:
    :param fig_size:
    :param title:
    :return:
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