import os
from pathlib import Path
from datetime import datetime
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
    def __init__(self, 
                 csv_path: Path, 
                 tables_dir: Path,
                 plot_dir: Path,
                 top_k_features: int = 400):
        self.csv_path = csv_path
        self.tables_dir = tables_dir
        self.plot_dir = plot_dir
        self.top_k_features = top_k_features
        self.models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "Ridge": Ridge(alpha=1.0),
            "Lasso": Lasso(alpha=0.1),
            "SVR": SVR(kernel='rbf', C=1.0, epsilon=0.1),
            "KNN": KNeighborsRegressor(n_neighbors=5)
        }
        self.predictions = {}

    def load_and_clean_data(self):
        self.df = pd.read_csv(self.csv_path, index_col=0)
        self.df.dropna(inplace=True)
        grab_col_names(self.df)

        self.y = self.df["log_price"]
        self.x = self.df.drop(columns="log_price")
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

            self.predictions[name] = y_pred

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

    def save_true_vs_predicted_table(self, model_name: str, predictions: np.ndarray):
        now = datetime.now().strftime("%d_%M%S")
 
        pred_df = pd.DataFrame({
            "true_log_price": self.y_test.values,
            "predicted_log_price": predictions,
            "residual": self.y_test.values - predictions
        })
        pred_df.index = self.df.index

        csv_path = self.tables_dir / f"true_vs_predicted_{model_name}_{now}.csv"
        pred_df.to_csv(csv_path, index=False)
        print(f"📄 Prediction table saved to: {csv_path}")


    def save_results_and_plot_best_model(self, results: dict):
        now = datetime.now().strftime("%d_%M%S")
        results_df = pd.DataFrame(results).T

        # Save CSV of metrics
        csv_path = self.tables_dir / f"model_metrics_{now}.csv"
        results_df.to_csv(csv_path)
        print(f"\n✅ Results saved to: {csv_path}")

        # Identify best model by lowest MSE
        best_model = min(results.items(), key=lambda x: x[1]["MSE"])[0]
        print(f"\n🏆 Best model: {best_model}")

        # Save true vs predicted table
        self.save_true_vs_predicted_table(best_model, self.predictions[best_model])

        # Plot true vs predicted
        plt.figure(figsize=(6, 6))
        sns.scatterplot(x=self.y_test, y=self.predictions[best_model], alpha=0.4)
        plt.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 '--', color='red')
        plt.xlabel("True log_price")
        plt.ylabel("Predicted log_price")
        plt.title(f"True vs Predicted - {best_model}")
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"true_vs_predicted_{best_model}_{now}.png")


    def run(self):
        self.load_and_clean_data()
        self.train_test_split()
        results = self.train_and_evaluate_models()
        self.plot_feature_importance()
        self.save_results_and_plot_best_model(results)
        return results


if __name__ == "__main__":
    base_path = Path(__file__).parents[1]
    # Get current date & format to YYYY-MM-DD
    now = datetime.now()
    current_date = now.strftime("%Y_%m_%d")
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
    file_path = results_dir / "intermediate" / "2025_04_11" / "train_test.csv"
    pipeline = PricePredictionPipeline(
        csv_path=file_path,
        tables_dir=tables_dir,
        plot_dir=plot_dir
    )
    results = pipeline.run()
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        print(f"Mean Squared Error: {metrics['MSE']:.2f}")
        print(f"R^2 Score: {metrics['R2']:.4f}")
