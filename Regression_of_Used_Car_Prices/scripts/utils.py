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
    return df

def _custom_transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Custom transformations on the DataFrame.
    
    :param df: DataFrame to be transformed.
    
    :return: Transformed DataFrame."""
    df["fuel_type"] = df["fuel_type"].str.replace("–", "U")
    # Format engine
    df["engine"] = df["engine"].str.replace("–", "U")
    # Format transmission
    df["transmission"] = df["transmission"].str.replace("–", "U")
    # Extract whether (A/T) vs (M/T) from the transmission column
    transmission_map = {
        "A/T": "Automatic",
        "M/T": "Manual"   
    }
    # Extract the substring (A/T) or (M/T) from the transmission column
    df["transmission_type"] = df["transmission"].str.extract(r'\b([AM]/T)\b', expand=False).map(transmission_map)
    df["transmission_type"] = df["transmission_type"].fillna("U")
    # Extract transmission speed ( X-Speed ) from the transmission column
    df["transmission_speed"] = df["transmission"].str.extract(r'(\d+)\s*-Speed', expand=False)
    # Convert the transmission speed to a float
    df["transmission_speed"] = df["transmission_speed"].astype(float)
    # Set all NaN values to 4 if transmission is either automatic or manual else 0
    df["transmission_speed"] = df["transmission_speed"].fillna(4)
    df["transmission_speed"] = df.apply(lambda x: 0 if x["transmission_type"] == "U" else x["transmission_speed"], axis=1)
    # Extract engine liter ( X.XL ) from the engine column 
    df["engine_liter"] = df["engine"].str.extract(r'(\d+(?:\.\d+)?)\s*L', expand=False)
    # Convert the engine liter to a float
    df["engine_liter"] = df["engine_liter"].astype(float)
    # Set all NaN values to 0
    df["engine_liter"] = df["engine_liter"].fillna(0)
    # We do the same for HP
    df["engine_hp"] = df["engine"].str.extract(r'(\d+(?:\.\d+)?)\s*HP', expand=False)
    # Convert the engine hp to a float
    df["engine_hp"] = df["engine_hp"].astype(float)
    # Set all NaN values to 0
    df["engine_hp"] = df["engine_hp"].fillna(0)
    # Extract number of cylinders
    df["engine_cylinders"] = df["engine"].str.extract(r'(\d+)\s*Cyl', expand=False)
    # Convert the engine cylinders to a float
    df["engine_cylinders"] = df["engine_cylinders"].astype(float)
    # Set all NaN values to 0
    df["engine_cylinders"] = df["engine_cylinders"].fillna(0)
    # Map accident history
    accident_map = {
        "None reported": False,
        "At least 1 accident or damage reported": True
    }
    df["accident"] = df["accident"].map(accident_map)
    # Convert the model year to a datetime object
    df["model_year"] = pd.to_datetime(df["model_year"], format="%Y")
    # sort the model year
    df = df.sort_values(by="model_year")
    # Convert the model year to a numeric value
    df["model_year"] = df["model_year"].dt.year
    # Drop columns
    df.drop(columns=["engine", "transmission", "clean_title", "ext_col", "int_col"], inplace=True)
    return df

def _log_transform_input_data(df: pd.DataFrame, vars: list = ["price", "engine_hp"]) -> pd.DataFrame:
    """
    Log transform the specified variables in the DataFrame.

    :param df: DataFrame with variables to be log transformed
    :param vars: List of variable names to be log transformed.

    :return: DataFrame with log transformed variables
    """
    # Log transform the specified variables
    for var in vars:
        if var in df.columns:
            # Apply log transformation
            df[var] = np.log1p(df[var])
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

def _impute_missing_data(
    df: pd.DataFrame, 
    target_var: str, 
    cat_feat: list,
    num_feat: list,
    strategy: dict = {
        "categorical": "simple",
        "numerical": "iterative"
    }
) -> pd.DataFrame:
    """Impute missing values in the DataFrame.
    
    :param df: DataFrame with missing values
    :param target_var: Target variable name.
    :param strategy: Imputation strategy for categorical and numerical variables.

    :return: Tuple of (transformer, DataFrame with imputed values)
    """
    print("Imputing missing values...")
    # Check if the target variable is in the DataFrame
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in DataFrame.")
    # Check if the categorical and numerical features are in the DataFrame
    for feature in cat_feat + num_feat:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' not found in DataFrame.")
    print(cat_feat, num_feat)

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
        cat_steps = [('imputer_cat', SimpleImputer(strategy='most_frequent'))]
        cat_pipe = Pipeline(steps=cat_steps)
    else:
        raise ValueError("Invalid categorical imputation method. Choose from 'simple'.")
    
    num_strategy = strategy["numerical"]
    num_steps = []
    if num_strategy == "simple":
        num_steps.append(('imputer_num', SimpleImputer(strategy='mean')))
    elif num_strategy == "knn":
        num_steps.append(('imputer_num', KNNImputer(n_neighbors=3)))
    elif num_strategy == "iterative":
        num_steps.append(('imputer_num', IterativeImputer(max_iter=10, random_state=0)))
    else:
        raise ValueError("""Invalid numerical imputation method. 
                         Choose from 'simple', 'knn', or 'iterative'.""")
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