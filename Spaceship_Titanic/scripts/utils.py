import pandas as pd
import numpy as np
from pathlib import Path
import datetime

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set scaler
scaler = StandardScaler()
# Set kmeans
k = 5  # You can experiment with different k values
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)

def _load_input_data(csv_path: Path) -> pd.DataFrame:
    """
    Load and preprocess the dataset from the given CSV path.

    :param csv_path: Path to the CSV file.

    :return: Preprocessed DataFrame.
    """
    # Load dataset
    df = pd.read_csv(csv_path, index_col=0)
    # Parse the passengerId column
    df[["passenger_group", "family_size"]] = df.reset_index()["PassengerId"].str.split("_", expand=True).values
    # Extract the family size and convert it to numeric
    df["family_size"] = df["family_size"].astype(np.int32)
    # Parse the Cabin column
    df[["deck", "deck_num", "deck_side"]] = df["Cabin"].str.split("/", expand=True)
    # Drop columns
    df.drop(columns=["Name", "Cabin", "passenger_group"], inplace=True)
    # Convert deck number to numeric
    df["deck_num"] = pd.to_numeric(df["deck_num"], errors="coerce")
    return df

def _custom_transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Custom transformations on the DataFrame.
    
    :param df: DataFrame to be transformed.
    
    :return: Transformed DataFrame."""
    # Create expanses column (sum of all expenses)
    df["Expanses"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]
    # Scale the spending data
    spending_data = df[df["Expanses"] > 0][["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]]
    spending_data_scaled = scaler.fit_transform(spending_data)
    # Cluster the spending data
    spending_clusters = kmeans.fit_predict(spending_data_scaled)
    df["Expanse_Cluster"] = -1
    df.loc[df["Expanses"] > 0, "Expanse_Cluster"] = spending_clusters
    # Convert the expanse cluster to categorical
    df["Expanse_Cluster"] = df["Expanse_Cluster"].astype(object)   
    print(df["Expanse_Cluster"].value_counts())
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