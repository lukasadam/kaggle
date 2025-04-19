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
from utils import (_load_input_data, 
                   _custom_transform_data,
                   _onehot_encode_input_data,
                   _impute_missing_data)

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (AdaBoostClassifier, 
                              RandomForestClassifier,
                              BaggingClassifier,
                              ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier)

from sklearn.linear_model import LogisticRegression
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import GridSearchCV

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


class TransportPredictionPipeline:
    """Pipeline for training and evaluating models for the Spaceship Titanic dataset."""
    def __init__(self, 
                 intermediate_dir: Path,
                 tables_dir: Path,
                 plot_dir: Path,
                 target_var: str = "Transported",
                 impute: str = "knn"):
        """Initialize the pipeline with directories for saving results and plots.
        
        :param tables_dir: Directory to save tables.
        :param plot_dir: Directory to save plots.
        :param target_var: Target variable for prediction.
        """
        # Set the directories for saving results and plots
        self.intermediate_dir = intermediate_dir
        self.tables_dir = tables_dir
        self.plot_dir = plot_dir
        # Initialize the models
        self.models = {
            "Nearest Neighbors": KNeighborsClassifier(3),
            "Linear SVM": SVC(kernel="linear", C=0.025, random_state=42),
            "RBF SVM": SVC(gamma=2, C=1, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
            "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
            "Logistic Regression": LogisticRegression(random_state=42),
            "XGBoost GBLinear": XGBClassifier(booster='gblinear', eval_metric='logloss', random_state=42),
            "XGBoost GBTrees": XGBClassifier(booster='gbtree', eval_metric='logloss', random_state=42),
            "LGBM": LGBMClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "Bagging": BaggingClassifier(random_state=42),
            "Extra Trees": ExtraTreesClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }
        # Create the pipeline
        self.pipeline = Pipeline([
            ('load', FunctionTransformer(func=_load_input_data)),
            ('custom', FunctionTransformer(func=_custom_transform_data))
            # Add more steps like imputing, encoding, modeling...
        ])
        # Set the scaler
        self.scaler = StandardScaler()
        # Set the stratified k-fold cross-validation
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # Set the target variable
        self.target_var = target_var
        # Set the dataframes for predictions
        self.predictions = {}

    def load_train_test(
        self, 
        train_csv_path: str, 
        test_csv_path: str
    ):
        """Load the training and test data from CSV files."""
        self.train_df = self.pipeline.transform(train_csv_path)
        print(self.train_df.head())
        # Set all categorical features
        self.cat_features = self.train_df.select_dtypes(include=[object]).columns.tolist()
        # Set all numerical features
        self.num_features = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        # train_df = train_df[["HomePlanet", "Destination", "Transported"]]
        self.test_df = self.pipeline.transform(test_csv_path)
    
    def prepare_data(
        self,
        one_hot_encode: bool = True,
        normalize: bool = True,
        strategy: dict = None,
        feature_selection: bool = False
    ):
        """Prepare the data for training and testing."""
        # Impute missing values (or drop)
        self.train_df = self.impute_missing_data(self.train_df, 
                                                 target_var=self.target_var, 
                                                 cat_features=self.cat_features,
                                                 num_features=self.num_features,
                                                 strategy=strategy)
        if one_hot_encode:
            # One-hot encode categorical variables
            self.train_df = self.encode_one_hot(self.train_df)
            self.test_df = self.encode_one_hot(self.test_df)
        if normalize:
            # Normalize input data
            self.train_df = self.normalize_input_data(self.train_df, train=True)
            self.test_df = self.normalize_input_data(self.test_df, train=False)
        if feature_selection:
            # Feature selection
            self.train_df = self.feature_selection(self.train_df, target_var=self.target_var)
            self.test_df = self.test_df[self.train_df.columns.tolist()]
        self.x_train = self.train_df.drop(columns=[self.target_var])
        self.x_test = self.test_df.copy()
        self.y_train = self.train_df[self.target_var]
        # Save final list of feature columns
        self.feature_columns = self.x_train.columns.tolist()

    def encode_one_hot(self, df: pd.DataFrame):
        # One-hot encode categorical variables
        df_onehot = _onehot_encode_input_data(df, self.cat_features)
        return df_onehot
    
    def normalize_input_data(self, df: pd.DataFrame, train=True):
        # Fit the scaler on the training data
        if train:
            # Fit the scaler on the training data
            self.scaler.fit(df[self.num_features])
        # Normalize the training data
        df[self.num_features] = self.scaler.transform(df[self.num_features])
        return df
    
    def impute_missing_data(self, df: pd.DataFrame, 
                            target_var: str, 
                            cat_features: list,
                            num_features: list,
                            strategy: dict = None):
        # Impute missing values 
        _, df = _impute_missing_data(df, 
                                     target_var=target_var, 
                                     cat_feat=cat_features,
                                     num_feat=num_features,
                                     strategy=strategy)
        return df
    
    def train_test_split(self):
        # Perform train-test split
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=0.2, random_state=42
        )
        self.sample_ids = self.x_val.index.tolist()
        print(f"Train shape: {self.x_train.shape}, Validation shape: {self.x_val.shape}")
        print(f"Train target shape: {self.y_train.shape}, Validation target shape: {self.y_val.shape}")
        # Print length of feature columns and sample ids
        print(f"Feature columns length: {len(self.feature_columns)}")
        print(f"Sample ids length: {len(self.sample_ids)}")
        # Save final x_train in intermediate directory
        self.x_train.to_csv(self.intermediate_dir / "x_train.csv", index=True)
        # Save final x_val in intermediate directory
        self.x_val.to_csv(self.intermediate_dir / "x_val.csv", index=True)
        # Save final x_test in intermediate directory
        self.x_test.to_csv(self.intermediate_dir / "x_test.csv", index=True)

    # TODO: Implement feature augmentation
    def feature_augmentation(self):
        pass
    
    # TODO: Implement feature selection
    def feature_selection(self):
        pass

    def train_and_evaluate_models(self):
        results = {}
        self.feature_importances = {}

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            # Perform straitified k-fold cross-validation
            cv_results = cross_val_score(model, self.x_train, self.y_train, cv=self.skf, scoring='accuracy')
            print(f"Cross-validation accuracy: {np.mean(cv_results):.4f} ± {np.std(cv_results):.4f}")
            # Fit the model on the training data (without cross-validation)
            model.fit(self.x_train, self.y_train)
            y_pred = model.predict(self.x_val)
            # Compute binary crossentropy (log loss)
            bce = log_loss(self.y_val, y_pred)
            # Compute accuracy
            accuracy = np.mean(y_pred == self.y_val)
            # Compute AUROC
            fpr, tpr, _ = roc_curve(self.y_val, y_pred)
            # Add results to dictionary
            results[name] = {"BCE": bce, 
                             "Accuracy": accuracy, 
                             "AUROC": auc(fpr, tpr),
                             "CV Accuracy": np.mean(cv_results),
                             "CV Std": np.std(cv_results)}
            self.predictions[name] = y_pred

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                self.feature_importances[name] = pd.Series(importances, index=self.x_train.columns).sort_values(ascending=False)
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
            "true_transport": self.y_val.values,
            "predicted_transport": predictions,
        }, index=self.sample_ids)

        csv_path = self.tables_dir / f"true_vs_predicted_{model_name}_{now}.csv"
        pred_df.to_csv(csv_path, index=True)
        print(f"📄 Prediction table saved to: {csv_path}")

    def save_results_and_plot_best_model(self, results: dict):
        now = datetime.now().strftime("%d_%M%S")
        results_df = pd.DataFrame(results).T

        # Save CSV of metrics
        csv_path = self.tables_dir / f"model_metrics_{now}.csv"
        results_df.to_csv(csv_path)
        print(f"\n✅ Results saved to: {csv_path}")

        # Identify best model by lowest MSE
        best_model = max(results.items(), key=lambda x: x[1]["AUROC"])[0]
        print(f"\n🏆 Best model: {best_model}")

        # Save true vs predicted table
        self.save_true_vs_predicted_table(best_model, self.predictions[best_model])

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(pd.crosstab(self.y_val, self.predictions[best_model], rownames=['True'], colnames=['Predicted']), annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {best_model}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"confusion_matrix_{best_model}_{now}.png")

        # Plot AUC-ROC curve
        fpr, tpr, _ = roc_curve(self.y_val, self.predictions[best_model])
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

    def grid_search_best_model(self, model_name: str):
        # Define the parameter grid for the model
        param_grid = {
            'n_estimators': list(range(50, 200, 50)),
            'max_depth': list(range(3, 10)),
            'learning_rate': list(np.arange(0.01, 0.1, 0.01)),
        }
        # Create the model
        model = self.models[model_name]
        # Create the grid search object
        grid_search = GridSearchCV(model, param_grid, cv=self.skf, scoring='accuracy', n_jobs=-1)
        # Fit the grid search
        grid_search.fit(self.x_train, self.y_train)
        # Get the best parameters
        best_params = grid_search.best_params_
        # Get the best score
        best_score = grid_search.best_score_
        # Get the best model
        best_model = grid_search.best_estimator_
        # Print the best parameters and score
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score:.4f}")   
        return best_model
    
    def make_predictions(self, model):
        # Make predictions on the test data
        y_pred = model.predict(self.x_test)
        # Save predictions to CSV
        pred_df = pd.DataFrame({
            "PassengerId": self.test_df.index,
            "Transported": y_pred
        })
        # Convert the transported predictions to boolean
        pred_df["Transported"] = pred_df["Transported"].astype(bool)
        # Save the predictions to a CSV file
        pred_csv_path = self.tables_dir / f"predictions_test_set.csv"
        pred_df.to_csv(pred_csv_path, index=False)
        print(f"Predictions saved to: {pred_csv_path}")
        return pred_df


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
    train_csv_path = data_dir / "train.csv"
    test_csv_path = data_dir / "test.csv"
    # Create the pipeline
    prediction_pipeline = TransportPredictionPipeline(
        intermediate_dir,
        tables_dir, 
        plot_dir,
        target_var="Transported"
    )
    prediction_pipeline.load_train_test(train_csv_path, test_csv_path)
    prediction_pipeline.prepare_data(
        one_hot_encode=True,
        normalize=True,
        strategy={
            "categorical": "simple",
            "numerical": "iterative"
        },
        feature_selection=False
    )
    prediction_pipeline.train_test_split()
    # Train and evaluate models
    results = prediction_pipeline.train_and_evaluate_models()
    for model_name, metrics in results.items():
        print(f"\nModel: {model_name}")
        for metric, value in metrics.items():
            if "CV" not in metric:
                print(f"{metric}: {value:.4f}")
    prediction_pipeline.save_results_and_plot_best_model(results)

    # Choose the best model based on the results (AUROC and perform grid search)
    #grid_search_best_model = prediction_pipeline.grid_search_best_model("LGBM")
    # Make predictions on the test data
    #predictions = prediction_pipeline.make_predictions(grid_search_best_model)
    #print(predictions.head())
