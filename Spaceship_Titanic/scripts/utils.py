import pandas as pd
import numpy as np
from pathlib import Path
import datetime

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def _load_input_data(csv_path: Path) -> pd.DataFrame:
    """
    Load and preprocess the dataset from the given CSV path.

    :param csv_path: Path to the CSV file.

    :return: Preprocessed DataFrame.
    """
    # Load dataset
    df = pd.read_csv(csv_path, index_col=0)
    # Parse the Cabin column
    df[["deck", "deck_num", "deck_side"]] = df["Cabin"].str.split("/", expand=True)
    # Drop columns
    df.drop(columns=["Name", "Cabin"], inplace=True)
    # Convert deck number to numeric
    df["deck_num"] = pd.to_numeric(df["deck_num"], errors="coerce")
    return df

def _custom_transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Custom transformations on the DataFrame.
    
    :param df: DataFrame to be transformed.
    
    :return: Transformed DataFrame."""
    # Create expanses column (sum of all expenses)
    df["Expanses"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
    # Log-transform all continuous variables except Age
    continuous_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    df[continuous_vars] = df[continuous_vars].apply(lambda x: np.log1p(x) if x.name != "Age" else x)
    return df

def _onehot_encode_input_data(df: pd.DataFrame, categorical_vars: list) -> pd.DataFrame:
    """
    One-hot encode the categorical variables in the DataFrame.

    :param df: DataFrame with categorical variables
    :param categorical_vars: List of categorical variable names to be one-hot encoded.

    :return: DataFrame with one-hot encoded categorical variables
    """
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=categorical_vars, drop_first=True, dtype='int')    
    return df

def _impute_missing_data(df: pd.DataFrame, target_var: str, impute="knn") -> pd.DataFrame:
    """Impute missing values in the DataFrame.
    
    :param df: DataFrame with missing values
    :param target_var: Target variable name.
    :param impute: Imputation strategy. Options are "simple", "knn", "iterative", or "drop".

    :return: Tuple of (transformer, DataFrame with imputed values)
    """
    # Extract the target variable
    target = df[target_var]
    # Drop the target variable from the DataFrame
    df = df.drop(columns=[target_var])
    # Get the list of continuous features
    num_feat = df.select_dtypes(include=[np.number]).columns.tolist()
    # Get the list of categorical features
    cat_feat = df.select_dtypes(include=[object]).columns.tolist()
    # If impute is "drop", simple drop rows with missing values
    if impute=="drop":
        # Drop rows with missing values
        transformer = None
        df_imputed = df.dropna()
        return transformer, df_imputed
    
    # Perform separate imputation for continuous and categorical features
    cat_steps = [('imputer_cat', SimpleImputer(strategy='most_frequent'))]
    cat_pipe = Pipeline(steps=cat_steps)

    num_steps = []
    if impute=="simple":
        num_steps.append(('imputer_num', SimpleImputer(strategy='mean')))
    elif impute=="knn":
        num_steps.append(('imputer_num', KNNImputer(n_neighbors=3)))
    elif impute=="iterative":
        num_steps.append(('imputer_num', IterativeImputer(max_iter=10, random_state=0)))
    else:
        raise ValueError("Invalid imputation method. Choose from 'simple', 'knn', 'iterative', or 'drop'.")
    num_pipe = Pipeline(steps=num_steps)

    # Column transformer
    transformer = ColumnTransformer(
        transformers=[
            ('num', num_pipe, num_feat),
            ('cat', cat_pipe, cat_feat)
        ]
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