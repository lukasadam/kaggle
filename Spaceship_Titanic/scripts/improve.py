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
from utils import parse_input_data

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Get current date & format to YYYY-MM-DD
now = datetime.datetime.now()
current_date = now.strftime("%Y_%m_%d")

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

base_path = Path(__file__).parents[1]
data_dir = base_path / "data"
csv_path = data_dir / "train.csv"

# Load true vs predicted values from model
pred_csv_path = "/Users/adaml9/Private/kaggle/Spaceship_Titanic/results/tables/2025_04_18/true_vs_predicted_Linear SVM_18_3710.csv"
pred_df = pd.read_csv(pred_csv_path, index_col=0)
# Subset to all passenger Ids that are wrongly predicted
pred_df = pred_df[pred_df["true_transport"]!=pred_df["predicted_transport"]]

parse_transformer = FunctionTransformer(func=parse_input_data)

pipeline = Pipeline([
    ('parser', parse_transformer),
    # Add more steps like imputing, encoding, modeling...
])

# Load the dataset
df = parse_transformer.transform(csv_path)

good_df = df.loc[list(set(df.index).difference(set(pred_df.index)))]
bad_df = df.loc[list(set(df.index).intersection(set(pred_df.index)))]