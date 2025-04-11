import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from utils import grab_col_names

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


class PricePredictionPipeline:
    def __init__(self, csv_path: Path, top_k_features: int = 400):
        self.csv_path = csv_path
        self.top_k_features = top_k_features
        self.models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
            "KNN": KNeighborsRegressor(n_neighbors=5)
        }

    def load_and_clean_data(self):
        df = pd.read_csv(self.csv_path, index_col=0)
        df.dropna(inplace=True)
        grab_col_names(df)

        self.y = df["log_price"]
        self.x = df.drop(columns="log_price")
        self.X_selected = self.x
        self.feature_names = self.X_selected.columns

    def train_test_split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.X_selected, self.y, test_size=0.2, random_state=42
        )

    def train_and_evaluate_models(self):
        results = {}
        self.feature_importances = {}

        for name, model in self.models.items():
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_test)
            mse = mean_squared_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            results[name] = {"MSE": mse, "R2": r2}

            # Save feature importances if available
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                self.feature_importances[name] = pd.Series(importances, index=self.feature_names).sort_values(ascending=False)

        return results

    def plot_feature_importance(self, top_n: int = 20):
        for model_name, importances in self.feature_importances.items():
            top_features = importances.head(top_n)
            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_features.values, y=top_features.index)
            plt.title(f"Top {top_n} Feature Importances - {model_name}")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.show()

    def run(self):
        self.load_and_clean_data()
        self.train_test_split()
        results = self.train_and_evaluate_models()
        self.plot_feature_importance()
        return results


if __name__ == "__main__":
    file_path = "/Users/adaml9/Private/kaggle/H&M_Scrapped_Product_Dataset_10_05_2025/results/intermediate/2025_04_11/train_test.csv"
    pipeline = PricePredictionPipeline(csv_path=file_path)    
    results = pipeline.run()
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"Mean Squared Error: {metrics['MSE']:.2f}")
        print(f"R^2 Score: {metrics['R2']:.4f}")
