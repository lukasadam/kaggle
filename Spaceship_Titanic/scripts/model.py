import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from utils import parse_input_data, onehot_encode_input_data

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


class TransportPredictionPipeline:
    def __init__(self, 
                 X: pd.DataFrame,
                 y: pd.Series,
                 tables_dir: Path,
                 plot_dir: Path):
        self.X = X.copy()
        self.y = y.copy()
        self.X_selected = self.X 
        self.feature_names = self.X_selected.columns
        self.tables_dir = tables_dir
        self.plot_dir = plot_dir
        self.models = {
            "Nearest Neighbors": KNeighborsClassifier(3),
            "Linear SVM": SVC(kernel="linear", C=0.025, random_state=42),
            "RBF SVM": SVC(gamma=2, C=1, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42)
        }
        self.predictions = {}

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
            
            # Compute binary crossentropy (log loss)
            bce = log_loss(self.y_test, y_pred)
            results[name] = {"BCE": bce}

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
            "true_transport": self.y_test.values,
            "predicted_transport": predictions,
        })

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
        best_model = min(results.items(), key=lambda x: x[1]["BCE"])[0]
        print(f"\n🏆 Best model: {best_model}")

        # Save true vs predicted table
        self.save_true_vs_predicted_table(best_model, self.predictions[best_model])

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.crosstab(self.y_test, self.predictions[best_model], rownames=['True'], colnames=['Predicted']), annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {best_model}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"confusion_matrix_{best_model}_{now}.png")

        # Plot AUC-ROC curve
        from sklearn.metrics import roc_curve, auc
        from sklearn.metrics import RocCurveDisplay
        fpr, tpr, _ = roc_curve(self.y_test, self.predictions[best_model])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {best_model}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"roc_curve_{best_model}_{now}.png")

    def run(self):
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
    # Set the path to the CSV file
    csv_path = data_dir / "train.csv"
    # Create the pipeline
    parse_transformer = FunctionTransformer(func=parse_input_data)
    pipeline = Pipeline([
        ('parser', parse_transformer),
        # Add more steps like imputing, encoding, modeling...
    ])
    # Load the dataset
    df = parse_transformer.transform(csv_path)
    # Target variable
    target_var = "Transported"
    # One-hot encode categorical variables
    categorical_vars = df.select_dtypes(include=[object]).columns.tolist()
    df_onehot = onehot_encode_input_data(df, categorical_vars)
    df_onehot = df_onehot.dropna()
    # Now we define X and y
    y = df_onehot[target_var]
    X = df_onehot.drop(columns=[target_var])
    
    prediction_pipeline = TransportPredictionPipeline(X, y, tables_dir, plot_dir)
    results = prediction_pipeline.run()
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
