import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from utils import _load_input_data, _custom_transform_data

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Get current date & format to YYYY-MM-DD
now = datetime.datetime.now()
current_date = now.strftime("%Y_%m_%d")

# Set the base path to the project directory
base_path = Path(__file__).parents[1]
# Set the path to the data directory
data_dir = base_path / "data"
results_dir = base_path / "results"
intermediate_dir = results_dir / "intermediate" / current_date
tables_dir = results_dir / "tables" / current_date
plot_dir = results_dir / "plots" / current_date
# Create directories if they don't exist
intermediate_dir.mkdir(parents=True, exist_ok=True)
tables_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)
# Set the path to the CSV file
data_dir = base_path / "data"
csv_path = data_dir / "train.csv"

# Specify transformers
load_transformer = FunctionTransformer(func=_load_input_data)

steps = [('load', load_transformer)]
pipeline = Pipeline(steps=steps)

# Load the dataset
df = pipeline.transform(csv_path)
# Target variable
target_var = "price"

# ============================================================= #
# We will clean the data & perform feature engineering

from utils import _custom_transform_data

# Custom transformations on the DataFrame
df = _custom_transform_data(df)

# ============================================================= #
# Exploratory Data Analysis (EDA) - Continuous and Categorical Variables

# Plot the distribution of the target variable (Price)
plt.figure(figsize=(10, 6))
sns.histplot(df[target_var], bins=50, kde=True)
plt.title("Distribution of Price")
plt.xlabel("Price")
plt.ylabel("Count")
plt.grid()
plt.savefig(plot_dir / "price_distribution.png")

# Log transform the target variable
df[target_var] = np.log1p(df[target_var])
# Plot the distribution of the log transformed target variable
plt.figure(figsize=(10, 6))
sns.histplot(df[target_var], bins=50, kde=True)
plt.title("Distribution of Log Transformed Price")
plt.xlabel("Log Transformed Price")
plt.ylabel("Count")
plt.grid()
plt.savefig(plot_dir / "log_transformed_price_distribution.png")

# Now we extract all continuous variables 
# and correlate them with each other 
# Get the list of continuous variables
continuous_vars = df.select_dtypes(include=[np.number]).columns.tolist()

# Make a sns pairplot of the continuous variables
sns.pairplot(df[continuous_vars], diag_kind='kde')
plt.savefig(plot_dir / "pairplot_continuous_variables.png")

# Log transform the engine hp
df["engine_hp"] = np.log1p(df["engine_hp"])
# Make a sns pairplot of the continuous variables
sns.pairplot(df[continuous_vars], diag_kind='kde')
plt.savefig(plot_dir / "pairplot_continuous_variables.png")

# Plot the distribution of the model year
plt.figure(figsize=(10, 6))
sns.histplot(df["model_year"], bins=50, kde=True)
plt.title("Distribution of Model Year")
plt.xlabel("Model Year")
plt.ylabel("Count")
plt.grid()
plt.savefig(plot_dir / "model_year_distribution.png")

# Plot the price by model year & rotate the x-axis labels
# order the dates
plt.figure(figsize=(10, 6))
sns.boxplot(x="model_year", y=target_var, data=df)
plt.title(f"Boxplot of {target_var} by Model Year")
plt.xlabel("Model Year")
plt.ylabel(target_var)
plt.xticks(rotation=90)
plt.grid()
plt.savefig(plot_dir / f"{target_var}_by_model_year.png")

