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
target_var = "Transported"

# ============================================================= #
# Exploratory Data Analysis (EDA) - Continuous and Categorical Variables

# Plot the distribution of the target variable (Transported)
plt.figure(figsize=(10, 6))
df[target_var].value_counts().plot(kind='bar')
plt.title("Distribution of Transported")
plt.xlabel("Transported")
plt.ylabel("Count")
plt.grid()
plt.savefig(plot_dir / "transported_distribution.png")

# Now we extract all continuous variables 
# and correlate them with each other 
# Get the list of continuous variables
continuous_vars = df.select_dtypes(include=[np.number]).columns.tolist()

# Make a sns pairplot of the continuous variables
sns.pairplot(df[continuous_vars], diag_kind='kde')
plt.savefig(plot_dir / "pairplot_continuous_variables.png")

# Observing that most variables are exponentially distributed 
# high zero inflation, we will log transform the variables
# except for age
df[continuous_vars] = df[continuous_vars].apply(lambda x: np.log1p(x) if x.name != "Age" else x)

sns.pairplot(df[continuous_vars], diag_kind='kde')
plt.savefig(plot_dir / "pairplot_continuous_variables_log.png")

# Now we will look at the categorical variables
# Get the list of categorical variables
categorical_vars = df.select_dtypes(include=[object]).columns.tolist()

# We will look at them individually and see how they relate with the target variable
plt.figure(figsize=(10, 6))
df.groupby(target_var)[categorical_vars[0]].value_counts().unstack().plot(kind='bar', stacked=True)
plt.title(f"Distribution of {categorical_vars[0]} by {target_var}")
plt.xlabel(categorical_vars[0])
plt.ylabel("Count")
plt.grid()
plt.savefig(plot_dir / f"{categorical_vars[0]}_by_{target_var}.png")

# We will do this for all categorical variables
for var in categorical_vars[1:]:
    plt.figure(figsize=(10, 6))
    df.groupby(target_var)[var].value_counts().unstack().plot(kind='bar', stacked=True)
    plt.title(f"Distribution of {var} by {target_var}")
    plt.xlabel(var)
    plt.ylabel("Count")
    plt.grid()
    plt.savefig(plot_dir / f"{var}_by_{target_var}.png")

# Plot cross tabulations of homeplanet and cryosleep
pd.crosstab(df["HomePlanet"], df["CryoSleep"]).plot(kind='bar', stacked=True)
pd.crosstab(df["HomePlanet"], df["Transported"]).plot(kind='bar', stacked=True)

# Now we will look at the correlation between the continuous variables and the target variable
# For that we will plot boxplots of the continuous variables
for var in continuous_vars:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=target_var, y=var, data=df)
    plt.title(f"Boxplot of {var} by {target_var}")
    plt.xlabel(target_var)
    plt.ylabel(var)
    plt.grid()
    plt.savefig(plot_dir / f"{var}_by_{target_var}.png")

# ============================================================= #
# Feature Engineering

# Specify transformers
load_transformer = FunctionTransformer(func=_load_input_data)

steps = [('load', load_transformer)]
pipeline = Pipeline(steps=steps)

# Load the dataset
df = pipeline.transform(csv_path)
# Target variable
target_var = "Transported"

# Create a expanses column
df["Expanses"] = df["RoomService"] + df["FoodCourt"] + df["ShoppingMall"] + df["Spa"] + df["VRDeck"]

# Plot the distribution of the expanses column
plt.figure(figsize=(10, 6))
df["Expanses"].plot(kind='hist', bins=50)
plt.title("Distribution of Expanses")
plt.xlabel("Expanses")
plt.ylabel("Count")
plt.grid()
plt.savefig(plot_dir / "expanses_distribution.png")

# Log transform the expanses column
df["Expanses"] = np.log1p(df["Expanses"])
# Plot the distribution of the log transformed expanses column
plt.figure(figsize=(10, 6))
df["Expanses"].plot(kind='hist', bins=50)
plt.title("Distribution of Log Transformed Expanses")
plt.xlabel("Log Transformed Expanses")
plt.ylabel("Count")
plt.grid()
plt.savefig(plot_dir / "log_transformed_expanses_distribution.png")

# Now we will look at the correlation between the expanses column and the target variable
plt.figure(figsize=(10, 6))
sns.boxplot(x=target_var, y="Expanses", data=df)
plt.title(f"Boxplot of Expanses by {target_var}")
plt.xlabel(target_var)
plt.ylabel("Expanses")
plt.grid()
plt.savefig(plot_dir / f"Expanses_by_{target_var}.png")

# ============================================================= #
# Imputation

from utils import _impute_missing_data

transformer, df_imputed = _impute_missing_data(df, impute="iterative")
