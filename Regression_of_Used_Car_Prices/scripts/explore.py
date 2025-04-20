import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime

import autogluon.eda.auto as auto
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
train_csv_path = data_dir / "train.csv"
test_csv_path = data_dir / "test.csv"
# Set the path to the AutoGluon output directory
autogluon_path = intermediate_dir / "autogluon_output" 

# Specify transformers
load_transformer = FunctionTransformer(func=_load_input_data)

steps = [('load', load_transformer)]
pipeline = Pipeline(steps=steps)

# Load the datasets
train_df = pipeline.transform(train_csv_path)
test_df = pipeline.transform(test_csv_path)
df = train_df.copy()
# Target variable
target_var = "price"

# ============================================================= #
# Obtain dataset overview

auto.dataset_overview(train_data=train_df, test_data=test_df, label=target_var)

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

# ============================================================= #
# Further manual exploration

# Now we extract all continuous variables 
# and correlate them with each other 
# Get the list of continuous variables
continuous_vars = df.select_dtypes(include=[np.number]).columns.tolist()

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

# Plot distribution of milage
plt.figure(figsize=(10, 6))
sns.histplot(df["milage"], bins=50, kde=True)
plt.title("Distribution of Milage")
plt.xlabel("Milage")
plt.ylabel("Count")
plt.grid()
plt.savefig(plot_dir / "milage_distribution.png")

# Log transform the milage
df["milage"] = np.log1p(df["milage"])
# Plot the distribution of the log transformed milage
plt.figure(figsize=(10, 6))
sns.histplot(df["milage"], bins=50, kde=True)
plt.title("Distribution of Log Transformed Milage")
plt.xlabel("Log Transformed Milage")
plt.ylabel("Count")
plt.grid()
plt.savefig(plot_dir / "log_transformed_milage_distribution.png")

# Plot distribution of engine hp
plt.figure(figsize=(10, 6))
sns.histplot(df["engine_hp"], bins=50, kde=True)
plt.title("Distribution of Engine HP")
plt.xlabel("Engine HP")
plt.ylabel("Count")
plt.grid()
plt.savefig(plot_dir / "engine_hp_distribution.png")

# Log transform the engine hp
df["engine_hp"] = np.log1p(df["engine_hp"])
# Plot the distribution of the log transformed engine hp
plt.figure(figsize=(10, 6))
sns.histplot(df["engine_hp"], bins=50, kde=True)
plt.title("Distribution of Log Transformed Engine HP")
plt.xlabel("Log Transformed Engine HP")
plt.ylabel("Count")
plt.grid()
plt.savefig(plot_dir / "log_transformed_engine_hp_distribution.png")

# ============================================================= #
# Automated quick fit using AutoGluon EDA

train_data = df
label = target_var
state = auto.quick_fit(
    df,
    label,
    path=autogluon_path,
    return_state=True,
    show_feature_importance_barplots=True
)

# ============================================================= #
# Target Variable Analysis using AutoGluon EDA

import autogluon.eda.analysis as eda
import autogluon.eda.visualization as viz
import autogluon.eda.auto as auto
from auto_utils import (_render_distribution_fit_information_if_available, 
                        _render_correlation_analysis,
                        _render_features_highly_correlated_with_target)

train_data = df
label = target_var
corr_threshold=0.1
fit_distributions = ['laplace_asymmetric', 'johnsonsu', 'exponnorm']

state = auto.analyze(
    train_data=train_data, 
    label=label,
    return_state=True,
    anlz_facets=[
    # Apply standard AutoGluon pre-processing to transform categorical variables to numbers to ensure correlation includes them.
    eda.transform.ApplyFeatureGenerator(category_to_numbers=True, children=[
        eda.dataset.DatasetSummary(),
        eda.missing.MissingValuesAnalysis(),
        eda.dataset.RawTypesAnalysis(),
        eda.dataset.SpecialTypesAnalysis(),
        eda.dataset.ProblemTypeControl(problem_type="auto")
    ])], 
    viz_facets=[
        viz.MarkdownSectionComponent("## Target variable analysis"),
        viz.dataset.DatasetStatistics(headers=True),
        viz.dataset.DatasetTypeMismatch(headers=True),
    ]
    )

state = auto.analyze_interaction(
        train_data=train_data,
        x=label,
        state=state,
        return_state=True,
        fit_distributions=fit_distributions
    )

state = _render_distribution_fit_information_if_available(state, label)
state = _render_correlation_analysis(state, train_data, label, corr_threshold=corr_threshold)
state = _render_features_highly_correlated_with_target(state, train_data, label)

# ============================================================= #
# Covariate Shift Analysis using AutoGluon EDA

import autogluon.eda.auto as auto

auto.covariate_shift_detection(train_data=train_df, test_data=test_df, label=target_var)

# ============================================================= #
# Feature Interaction Analysis using AutoGluon EDA

import autogluon.eda.auto as auto

auto.partial_dependence_plots(df, 
                              label=target_var, 
                              path=autogluon_path,
                              features=['model_year', 'accident'], two_way=True)

# ============================================================= #
# Anomaly Detection Analysis using AutoGluon EDA

threshold_stds = 3

state = auto.detect_anomalies(
    train_data=df, 
    label=target_var,
    threshold_stds=threshold_stds,
    show_top_n_anomalies=5,
    explain_top_n_anomalies=5,
    return_state=True,
    fig_args={
        'figsize': (6, 4)
    },
    chart_args={
        'normal.color': 'lightgrey',
        'anomaly.color': 'orange',
    }
)

train_anomaly_scores = state.anomaly_detection.scores.train_data

auto.analyze_interaction(train_data=df.join(train_anomaly_scores), 
                         x="model_year", y="price", 
                         hue="score", chart_args=dict(palette='viridis'))