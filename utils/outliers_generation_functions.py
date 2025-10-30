import time

import numpy as np
import pandas as pd
import numbers
import os
import warnings
from pandas import Series
from utils import helper_functions as hf


###############################################
# PERFTUBATE ORIGINAL DATA TO CREATE OUTLIERS #
###############################################
def perturb_within_distribution(
        original_data: pd.DataFrame,
        pct_to_perturb: int,
        target_column: str,
        feats_to_perturb: list[str] | int = None,
        cutoff_point: int = None,
        excluded_columns: list[str] = None,
        proportional: bool = True,
        gamma: float = 2.0,
        random_seed: int = None,
        decimal_places = 2,
        negative_values: bool = True,
        save: bool = False,
        directory_name: str = 'unnamed',
        key_word: str = ''
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Adds specific noise to the original data to create new outliers.
    The resulting CSV file is saved in the `data/perturbed_datasets/{directory_name}` directory.

    Args:
        original_data (pd.DataFrame): DataFrame containing the original data to modify.
        pct_to_perturb (float): Percentage of original rows to perturb.
        target_column (str): Name of the column containing class labels.

        feats_to_perturb (list[str] | int, optional): List of feature names to modify,
            or an integer specifying the number of random features to modify per row.
            Default behavior randomly modifies a random number of features per row.
            If left as default, pay attention to the `cutoff_point` parameter.
        cutoff_point (int, optional): Maximum number of features to randomly modify per row.
            Used only when `feats_to_perturb` is left as default.
        excluded_columns (list[str], optional): Columns to exclude from the perturbation process.
        proportional (bool, optional): If True, applies row perturbations based on the class label distribution.
        gamma (float, optional): Multiplier for the standard deviation that defines the scale of the noise.
        random_seed (int, optional): Seed for the random number generator.
        decimal_places (int, optional): Number of decimal places in modified data.
            Tip: Set to the maximum number of decimal places across all features.
        negative_values (bool, optional): Whether to allow negative values in the output data.
        save (bool, optional): Whether to save the perturbed dataset as a CSV file.
        directory_name (str, optional): Subdirectory name under `data/perturbed_datasets/` to save the file.
        key_word (str, optional): Keyword used to identify the output CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing modified rows (with original row indexes).
    """

    if random_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(random_seed)

    # Input check
    if pct_to_perturb < 1:
        raise ValueError("Percentage value must be positive integer.")
    else:
        pct_to_perturb *= 0.01

    if gamma < 0:
        raise ValueError("gamma must be positive float number.")

    if target_column not in original_data.columns:
        raise KeyError(f"There is no column named '{target_column}' in this DataFrame. ")

    if excluded_columns is None:
        excluded_columns = []

    if 'target' not in excluded_columns:
        excluded_columns.append('target')

    # Parameters
    modified_data: pd.DataFrame = original_data.copy()
    modified_data.rename(columns={target_column: 'target'}, inplace=True)

    features_stds: np.ndarray = modified_data.drop(columns='target').std().to_numpy()
    features_mins: np.ndarray = modified_data.drop(columns='target').min().to_numpy()
    all_features: list = [f for f in modified_data.columns
                    if f not in excluded_columns]
    n_rows: int = len(modified_data)
    n_cols: int = len(all_features)

    num_to_perturb: int = int(pct_to_perturb * n_rows + 0.5)

    if proportional:
        proportions: Series = modified_data['target'].value_counts(normalize=True)
        num_per_class: list = hf.proportional_split(num_to_perturb, proportions.values)
        num_per_class: dict = dict(zip(proportions.index, num_per_class))

        modified_idxs: list = []

        for (label, num) in num_per_class.items():
            label_idxs: np.ndarray = (
                modified_data[modified_data['target'] == label].index.to_numpy())
            selected_idxs: np.ndarray = rng.choice(label_idxs, size=num, replace=False)

            modified_idxs.extend(selected_idxs)
    else:
        modified_idxs: np.ndarray = rng.choice(n_rows, num_to_perturb, replace=False)

    # Each row modified independently
    for i in modified_idxs:
        # Features selected by user
        if isinstance(feats_to_perturb, list):
            if len(feats_to_perturb) == 0:
                raise ValueError("List of features to perturb cannot be empty.")

            cols_to_perturb: list = []
            invalid_features: list = []

            for f in feats_to_perturb:
                if f in all_features:
                    cols_to_perturb.append(f)
                else:
                    invalid_features.append(f)

            if invalid_features:
                warnings.warn(f"Invalid features ignored: {invalid_features}")

            if not cols_to_perturb:
                raise ValueError("No valid features to perturb provided in the list.")

        # Features selected randomly
        else:
            # Integer chosen by user
            if isinstance(feats_to_perturb, (int, np.integer)):
                k: int = feats_to_perturb

                if k > n_cols:
                    k = n_cols
                    warnings.warn(
                        f"Parameter feats_to_perturb exceeds the number of features. Clamped to {n_cols}.")

                if k <= 0:
                    raise ValueError("No valid features to perturb provided in the list.")

            # Random integer to be generated
            else:
                if cutoff_point is None:
                    cutoff_point = n_cols

                if  cutoff_point > n_cols or cutoff_point <= 0:
                    cutoff_point = n_cols
                    warnings.warn(
                        "Invalid cutoff_point parameter. No additional restriction has been applied.")

                k: int = rng.integers(1, cutoff_point + 1)

            # List of fetures to be randomly selected
            cols_to_perturb: list = list(rng.choice(all_features, size=k, replace=False))

        # Select features to be modified in one row
        col_idx: np.ndarray = modified_data.columns.get_indexer(cols_to_perturb)
        original_cells: np.ndarray = modified_data.iloc[i, col_idx].to_numpy()
        stds: np.ndarray = features_stds[col_idx]
        noise: np.ndarray = rng.standard_normal(size=original_cells.shape)*(stds*gamma)

        perturbed_cells: np.ndarray = original_cells + noise

        if not negative_values:
            for idx in np.ndindex(perturbed_cells.shape):
                value = perturbed_cells[idx]
                if value < 0:
                    feature_min: float = float(features_mins[col_idx[idx[0]]])
                    perturbed_cells[idx] = rng.uniform(0, 0.25 * feature_min)

        perturbed_cells = np.round(perturbed_cells, decimal_places)

        # Assign values to DataFrame
        modified_data.iloc[i, col_idx] = perturbed_cells

    if isinstance(feats_to_perturb, list):
        count = len(feats_to_perturb)
    elif isinstance(feats_to_perturb, (int, np.integer)):
        count = k
    else:
        count = f"rnd_{int(time.time() * 1000)}"

    if save:
        folder_path = f'../../data/perturbed_datasets/{directory_name}'
        os.makedirs(folder_path, exist_ok=True)

        filename = f'../../data/perturbed_datasets/{directory_name}/{key_word}_P[{round(pct_to_perturb*100)}]_F[{count}]_G[{gamma}].csv'
        modified_data.to_csv(filename, index=False)

    return modified_data, modified_idxs