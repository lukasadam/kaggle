import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime

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
results_dir = base_path / "results"

# Load true vs predicted values from model
pred_csv_path = results_dir / "tables" / "2025_04_20" / "true_vs_predicted_XGBoost GBTrees_20_5342.csv"
pred_orig_df = pd.read_csv(pred_csv_path, index_col=0)
# Load the original x_val data
val_csv_path = results_dir / "intermediate" / "2025_04_20" / "x_val.csv"
df = pd.read_csv(val_csv_path, index_col=0)

df = pd.merge(df, pred_orig_df, left_index=True, right_index=True, how='inner')
df["diff"] = df["predicted_transport"] - df["true_transport"]
# Plot the feature correlation matrix for the good and bad predictions
def plot_correlation_matrix(df, title, save_path):
    plt.figure(figsize=(20, 20))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Plot the correlation matrix for good predictions
plot_correlation_matrix(good_df, "Good Predictions Correlation Matrix", plot_dir / "good_predictions_correlation_matrix.png")
# Plot the correlation matrix for bad predictions
plot_correlation_matrix(bad_df, "Bad Predictions Correlation Matrix", plot_dir / "bad_predictions_correlation_matrix.png")

mean_diff = bad_df.mean(numeric_only=True) - good_df.mean(numeric_only=True)
mean_diff = mean_diff.sort_values(key=abs, ascending=False)

# Plot
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=mean_diff.values, y=mean_diff.index)
plt.title("Feature mean differences (Wrong - Correct)")
plt.xlabel("Mean Difference")
plt.tight_layout()
plt.show()

# Plot PCA for good and bad predictions (join both, and color by prediction)
from sklearn.decomposition import PCA
# Concatenate the good and bad dataframes
combined_df = pd.concat([good_df, bad_df])
# Perform PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(combined_df.drop(columns=['true_transport', 'predicted_transport']))
# Create a DataFrame with PCA results
pca_df = pd.DataFrame(data=pca_result, columns=['PCA1', 'PCA2'])
# Add the true transport and predicted transport columns
pca_df['true_transport'] = combined_df['true_transport'].values
pca_df['predicted_transport'] = combined_df['predicted_transport'].values
# Add the passenger Ids
pca_df['PassengerId'] = combined_df.index.values
# Add the prediction correctness
pca_df['prediction_correct'] = pca_df['true_transport'] == pca_df['predicted_transport']
# Plot PCA results
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='prediction_correct', palette=['red', 'green'])
plt.title("PCA of Good and Bad Predictions")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(title='Prediction Correctness', loc='upper right')
plt.tight_layout()
plt.savefig(plot_dir / "pca_good_bad_predictions.png")

# Plot true transport on PCA
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='true_transport', palette='Set1')
plt.title("PCA of True Transport")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend(title='True Transport', loc='upper right')
plt.tight_layout()
plt.savefig(plot_dir / "pca_true_transport.png")