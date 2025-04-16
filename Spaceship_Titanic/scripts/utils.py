import pandas as pd
import numpy as np
from pathlib import Path
import datetime

def parse_input_data(csv_path: Path) -> pd.DataFrame:
    """
    Load and preprocess the dataset from the given CSV path.

    Args:
        csv_path (Path): Path to the CSV file.
        target_var (str): Name of the target variable.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    # Load dataset
    df = pd.read_csv(csv_path, index_col=0)

    # Parse the Cabin column
    df[["deck", "deck_num", "deck_side"]] = df["Cabin"].str.split("/", expand=True)

    # Drop columns
    df.drop(columns=["Name", "Cabin"], inplace=True)

    # Convert deck number to numeric
    df["deck_num"] = pd.to_numeric(df["deck_num"], errors="coerce")

    # Log-transform all continuous variables except Age
    continuous_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    df[continuous_vars] = df[continuous_vars].apply(lambda x: np.log1p(x) if x.name != "Age" else x)

    return df

def onehot_encode_input_data(df: pd.DataFrame, categorical_vars: list) -> pd.DataFrame:
    """
    One-hot encode the categorical variables in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        categorical_vars (list): List of categorical variable names.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded categorical variables.
    """
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=categorical_vars, drop_first=True)
    
    return df