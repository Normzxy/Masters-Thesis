import numpy as np
import pandas as pd
import os
import warnings
from utils import helper_functions as hf

###############################################
# PERFTUBATE ORIGINAL DATA TO CREATE OUTLIERS #
###############################################
def perturb_within_distribution(
        original_data: pd.DataFrame,
        pct_to_perturb: int,
        target_column: str,
        features_to_perturb: list[str] | int = -1,
        excluded_columns: list[str] = None,
        proportional_perturbation: bool = True,
        sigma: float = 2.0,
        random_state: int = 42,
        decimal_places = 2,
        negative_values: bool = True,
        save: bool = False,
        directory_name: str = 'unnamed',
        key_word: str = ''
) -> tuple[pd.DataFrame, np.array]:
    """
    Adds a specific noise to the original data to create new outliers.
    CSV file is then saved in a data/perturbed_datasets/{directory_name} directory.

    OBLIGATORY
    :param original_data: DataFrame containing original data to modify.
    :param pct_to_perturb: Percentage (int) of original rows to perturb.
    :param target_column: The name of the column containing the class labels.

    OPTIONAL
    :param features_to_perturb: List of feature names to modify or integer for number of features to randmly modify per row.
    :param excluded_columns: List of column names to exclude from a perturbation process.
        If neither list nor non-positive integer, random modify count per row.
    :param proportional_perturbation: If True, applies row perturbations based on the class label distribution.
    :param sigma: Amount multiplied by the standard deviation. Defines the scale of the noise.
    :param random_state: Seed for random number generator.
    :param decimal_places: Number of decimal places to use in modified data.
        Advice: Set the value to the maximum number of decimal places across the whole df.
    :param negative_values: Determines whether the output data should always be positive.
    :param save: Whether to save the new CSV file.
    :param directory_name: Files are saved in data/perturbed_datasets/{directory_name}.
    :param key_word: Keyword to use to identify a new CSV file.
    :return: DataFrame with certain rows modified with rows indexes.
    """

    rng = np.random.default_rng(random_state)

    # Input check
    if pct_to_perturb < 1:
        raise ValueError("Percentage value must be positive integer.")
    else:
        pct_to_perturb *= 0.01

    if sigma < 0:
        raise ValueError("Sigma must be positive float number.")

    if target_column not in original_data.columns:
        raise KeyError(f"There is no {target_column} column in this DataFrame. Mark label column as 'target' first.")

    if excluded_columns is None:
        excluded_columns = []
    if 'target' not in excluded_columns:
        excluded_columns.append('target')

    # Parameters
    modified_data = original_data.copy()
    modified_data.rename(columns={target_column: 'target'}, inplace=True)
    all_features = [f for f in modified_data.columns
                    if f not in excluded_columns]
    n_rows = len(modified_data)
    n_cols = len(all_features)
    all_stds = modified_data.std().to_numpy()
    mins = modified_data.min().to_numpy()

    num_to_perturb = int(pct_to_perturb * n_rows)

    if proportional_perturbation:
        proportions = modified_data['target'].value_counts(normalize=True)
        num_per_class = hf.proportional_split(num_to_perturb, proportions.values)
        num_per_class = dict(zip(proportions.index, num_per_class))

        modified_idxs = []
        for label, num in num_per_class.items():
            label_idxs = (
                modified_data[modified_data['target'] == label].index.to_numpy())
            selected_idxs = rng.choice(label_idxs, size=num, replace=False)
            modified_idxs.extend(selected_idxs)
    else:
        modified_idxs = rng.choice(n_rows, num_to_perturb, replace=False)

    # Each row modified independently
    for i in modified_idxs:
        # Features selected by user
        if isinstance(features_to_perturb, list) and features_to_perturb:
            cols_to_perturb = []
            invalid_features = []

            for f in features_to_perturb:
                if f in all_features:
                    cols_to_perturb.append(f)
                else:
                    invalid_features.append(f)

            if invalid_features:
                warnings.warn(f"Invalid features: {invalid_features}")

        # Features selected randomly
        else:
            # Integer chosen by user
            if isinstance(features_to_perturb, int) and (features_to_perturb > 0):
                k = min(features_to_perturb, n_cols)
                if features_to_perturb > n_cols:
                    warnings.warn(f"Chosen value exceeds the number of features. Clamped to {n_cols}.")
            # Random integer to be generated
            else:
                k = rng.integers(1, n_cols + 1)

            # List of fetures to be randomly selected
            cols_to_perturb = list(rng.choice(all_features, size=k, replace=False))

        # Select features to be modified in one row
        col_idx = modified_data.columns.get_indexer(cols_to_perturb)
        original_cells = modified_data.iloc[i, col_idx].to_numpy()
        stds = all_stds[col_idx]
        noise = rng.standard_normal(size=original_cells.shape)*(stds*sigma)

        perturbed_cells = original_cells + noise
        if not negative_values:
            # Replace any negative values by sampling from [0, 0.25 * min_feature_value]
            for j, value in enumerate(perturbed_cells):
                if value < 0:
                    feature_min = mins[col_idx[j]]
                    perturbed_cells[j] = rng.uniform(0, 0.25 * feature_min)
        perturbed_cells = np.round(perturbed_cells, decimal_places)

        # Assign values to DataFrame
        modified_data.iloc[i, col_idx] = perturbed_cells

    if save:
        folder_path = f'../../data/perturbed_datasets/{directory_name}'
        os.makedirs(folder_path, exist_ok=True)
        filename = f'../../data/perturbed_datasets/{directory_name}/{key_word}_{round(pct_to_perturb*100)}_pct.csv'
        modified_data.to_csv(filename, index=False)

    return modified_data, modified_idxs


################################################
# GENERATE AROUND EXISTING OUTLIERS WITH NOISE #
################################################
def generate_around_outliers(
    original_data: pd.DataFrame,
    input_outliers: pd.DataFrame,
    pct_to_enter: int,
    feature_range_pct: int = 33,
    random_state: int = 42,
    decimal_places = 2,
    negative_values: bool = False,
    save: bool = False,
    directory_name: str = 'unnamed',
    key_word: str = ''
) -> pd.DataFrame:
    """
    Generates a new CSV file consisting of new outliers, based on original data.
    The generated values are based on the values given to the function.
    CSV file is then saved in a data/generated_data/{directory_name} directory.
    Outlier Y label should be named 'target'.

    OBLIGATORY
    :param original_data: DataFrame containing original data.
    :param input_outliers: Reference outliers to generate around.
    :param pct_to_enter: Percentage (int) of original rows to enter.

    OPTIONAL
    :param feature_range_pct: Percentage (int) to calculate sigma,
        where sigma is a value representing some percentage (sigma_frac) of each feature range.
    :param random_state: Seed for random number generator.
    :param decimal_places: Number of decimal places to use in new data.
        Advice: Set the value to the maximum number of decimal places across the whole df.
    :param negative_values: Determines whether the output data should always be positive.
    :param save: Whether to save the new CSV file.
    :param directory_name: Files are saved in data/generated_data/{directory_name}.
    :param key_word: Keyword to use to identify a new CSV file.
    :return: DataFrame containing new outliers.
    """

    rng = np.random.default_rng(random_state)

    if feature_range_pct < 1 and pct_to_enter < 1:
        raise ValueError("Percentage values must be positive integers.")
    else:
        feature_range_pct *= 0.01
        pct_to_enter *= 0.01

    num_to_generate = int(pct_to_enter * len(original_data))
    feature_range = original_data.max(axis=0) - original_data.min(axis=0)
    sigma = (feature_range_pct*feature_range).to_numpy()

    bases = (
        input_outliers
        .sample(n=num_to_generate, replace=True, random_state=rng)
        .reset_index(drop=True)
    )

    features = original_data.columns.difference(['target'])
    min_vals = input_outliers[features].min()*feature_range_pct
    lower_noise = min_vals.values - bases[features].to_numpy()

    noise = pd.DataFrame(
        rng.standard_normal(size=(
            num_to_generate, len(features)))*sigma,
            columns=features,
            index=bases.index)

    if not negative_values:
        noise = noise.clip(lower=lower_noise)

    new_data = (bases[features] + noise).round(decimal_places)
    new_data['target'] = bases['target'].values

    if save:
        folder_path = f'../../data/generated_around_outliers/{directory_name}'
        os.makedirs(folder_path, exist_ok=True)
        filename = f'../../data/generated_around_outliers/{directory_name}/{key_word}_{round(pct_to_enter*100)}_pct.csv'
        new_data.to_csv(filename, index=False)

    return new_data