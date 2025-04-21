from typing import List, Optional, Dict

import pandas as pd
import numpy as np
from pathlib import Path
import datetime
from pandas import DataFrame
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error


def _load_input_data(csv_path: Path) -> DataFrame:
    """Load and preprocess the dataset from the given CSV path.

    :param csv_path: Path to the CSV file.
    :type csv_path: Path
    :return: DataFrame containing the loaded data.
    :rtype: DataFrame
    """
    # Load dataset
    df = pd.read_csv(csv_path, index_col=0)
    return df


def _custom_transform_data(df: DataFrame) -> DataFrame:
    """Custom transformations on the DataFrame.

    :param df: DataFrame to be transformed.
    :type df: DataFrame
    :return: Transformed DataFrame.
    :rtype: DataFrame
    """
    # Fill missing values
    for feature in ["fuel_type", "accident", "clean_title"]:
        df[feature] = df[feature].fillna("U")
    # Clean the title column
    df["clean_title"] = np.where(df["clean_title"] == "Yes", 1, 0).astype("int16")
    # Replace '-' with 'not supported' in fuel_type column
    df["fuel_type"] = df["fuel_type"].replace({"–": "not supported"})
    # Format engine
    df["engine"] = df["engine"].str.replace("–", "U")
    # Format transmission
    df["transmission"] = df["transmission"].str.replace("–", "U")
    # Define regex pattern to match A/T, M/T, automatic, manual (case-insensitive)
    pattern = r"\b(A/T|M/T|automatic|manual)\b"
    # Extract and normalize
    df["transmission_type"] = (
        df["transmission"].str.extract(pattern, expand=False).str.lower()
    )
    # Optional: map to friendly labels
    transmission_map = {
        "a/t": "Automatic",
        "automatic": "Automatic",
        "m/t": "Manual",
        "manual": "Manual",
    }
    df["transmission_type"] = df["transmission_type"].map(transmission_map)
    df["transmission_type"] = df["transmission_type"].fillna("U")
    # Extract transmission speed ( X-Speed ) from the transmission column
    df["transmission_speed"] = df["transmission"].str.extract(
        r"(\d+)\s*-Speed", expand=False
    )
    # Convert the transmission speed to a categorical
    df["transmission_speed"] = df["transmission_speed"].astype(object)
    # Set all NaN values to 4 if transmission is either automatic or manual else -1
    df["transmission_speed"] = df["transmission_speed"].fillna("U")
    # Extract engine liter ( X.XL ) from the engine column
    df["displacement"] = df["engine"].str.extract(r"(\d+(?:\.\d+)?)\s*L", expand=False)
    # Convert the engine liter to a float
    df["displacement"] = df["displacement"].astype(float)
    # Set all NaN values to U
    df["displacement"] = df["displacement"].fillna(0)
    # We do the same for HP
    df["engine_hp"] = df["engine"].str.extract(r"(\d+(?:\.\d+)?)\s*HP", expand=False)
    # Convert the engine hp to a float
    df["engine_hp"] = df["engine_hp"].astype(float)
    # Set all NaN values to 0
    df["engine_hp"] = df["engine_hp"].fillna(0)
    #  Extract whether GDI/PDI/Turbo from the engine column
    df["engine_gdi"] = df["engine"].str.extract(
        r"(\bGDI\b|\bPDI\b|\bTurbo\b)", expand=False
    )
    df["engine_gdi"] = df["engine_gdi"].fillna("U")
    # Extract number of cylinders
    df["engine_cylinders"] = df["engine"].str.extract(r"(\d+)\s*Cyl", expand=False)
    # Convert the engine cylinders to a categorical
    df["engine_cylinders"] = df["engine_cylinders"].astype(object)
    # Set all NaN values to U
    df["engine_cylinders"] = df["engine_cylinders"].fillna("U")
    # Convert the model year to a datetime object
    df["model_year"] = pd.to_datetime(df["model_year"], format="%Y")
    # sort the model year
    df = df.sort_values(by="model_year")
    # Convert the model year to a numeric value
    df["model_year"] = df["model_year"].dt.year
    # Drop columns
    df.drop(columns=["engine", "transmission", "clean_title"], inplace=True)
    return df


def _bin_price(data: DataFrame) -> DataFrame:
    """Bin the price variable into 0 and 1 based on the upper bound.

    :param data: DataFrame with the price variable.
    :type data: DataFrame
    :return: DataFrame with the binned price variable.
    :rtype: DataFrame
    """
    # Make copy of the DataFrame
    df = data.copy()
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = np.percentile(df["price"], 25)
    Q3 = np.percentile(df["price"], 75)
    IQR = Q3 - Q1
    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Identify outliers
    outliers = df[(df["price"] > upper_bound)]
    df["price"] = (df["price"] < upper_bound).astype(int)
    return df


def _log_transform_input_data(
    df: DataFrame, vars: List[str] = ["milage", "price", "engine_hp"]
) -> DataFrame:
    """Log transform the specified variables in the DataFrame

    :param df: DataFrame with variables to be log transformed
    :type df: DataFrame
    :param vars: List of variable names to be log transformed, defaults to ["milage", "price", "engine_hp"]
    :type vars: List[str], optional
    :return: DataFrame with log transformed variables
    :rtype: DataFrame
    """
    # Log transform the specified variables
    for var in vars:
        if var in df.columns:
            # Apply log transformation
            df[var] = np.log1p(df[var])
    return df


def _onehot_encode_input_data(df: DataFrame, categorical_vars: List[str]) -> DataFrame:
    """_summary_

    :param df: One-hot encode the categorical variables in the DataFrame.
    :type df: DataFrame
    :param categorical_vars: List of categorical variable names to be one-hot encoded.
    :type categorical_vars: list
    :return: DataFrame with one-hot encoded categorical variables
    :rtype: DataFrame
    """
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=categorical_vars, drop_first=True, dtype="int")
    return df


def _impute_missing_data(
    df: DataFrame,
    target_var: str,
    cat_feat: List[str],
    num_feat: List[str],
    strategy: Optional[Dict[str, str]] = {"categorical": "simple", "numerical": "knn"},
) -> tuple[ColumnTransformer, DataFrame]:
    """Function to impute missing values in the DataFrame.

    :param df: DataFrame to be imputed
    :type df: DataFrame
    :param target_var: Target variable to be predicted
    :type target_var: str
    :param cat_feat: Categorical features to be imputed
    :type cat_feat: list
    :param num_feat: Numerical features to be imputed
    :type num_feat: list
    :param strategy: Strategy for imputation, defaults to {"categorical": "simple", "numerical": "knn"}
    :type strategy: dict, optional
    :return: Imputed DataFrame
    :rtype: DataFrame
    """
    # Check if the target variable is in the DataFrame
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in DataFrame.")
    # Check if the categorical and numerical features are in the DataFrame
    for feature in cat_feat + num_feat:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame.")
    print(f"Imputing missing values for {cat_feat} and {num_feat}...")
    # Extract the target variable
    target = df[target_var]
    # Drop the target variable from the DataFrame
    df = df.drop(columns=[target_var])
    # If imputation strategy is None, simple drop rows with missing values
    if strategy is None:
        # Drop rows with missing values
        transformer = None
        df_imputed = df.dropna()
        return transformer, df_imputed
    cat_strategy = strategy["categorical"]
    if cat_strategy == "simple":
        # Perform separate imputation for continuous and categorical features
        cat_steps = [("imputer_cat", SimpleImputer(strategy="most_frequent"))]
        cat_pipe = Pipeline(steps=cat_steps)
    else:
        raise ValueError("Invalid categorical imputation method. Choose from 'simple'.")
    num_strategy = strategy["numerical"]
    num_steps = []
    if num_strategy == "simple":
        num_steps.append(("imputer_num", SimpleImputer(strategy="mean")))
    elif num_strategy == "knn":
        num_steps.append(("imputer_num", KNNImputer(n_neighbors=3)))
    elif num_strategy == "iterative":
        num_steps.append(("imputer_num", IterativeImputer(max_iter=10, random_state=0)))
    else:
        raise ValueError(
            """Invalid numerical imputation method. 
                         Choose from 'simple', 'knn', or 'iterative'."""
        )
    num_pipe = Pipeline(steps=num_steps)
    # Column transformer
    transformer = ColumnTransformer(
        transformers=[("num", num_pipe, num_feat), ("cat", cat_pipe, cat_feat)]
    )
    # Fit the transformer
    transformer.fit(df)
    # Transform the data
    df_imputed = transformer.transform(df)
    # Convert the transformed data back to a DataFrame
    df_imputed = pd.DataFrame(df_imputed, columns=num_feat + cat_feat, index=df.index)
    # Add the target variable back to the DataFrame
    df_imputed[target_var] = target
    return transformer, df_imputed


def _exp1m_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the root mean squared error (RMSE) between the true and predicted values.

    :param y_true: True values
    :type y_true: np.ndarray
    :param y_pred: Predicted values
    :type y_pred: np.ndarray
    :return: RMSE value
    :rtype: np.ndarray
    """
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true_exp, y_pred_exp))
    return -rmse
