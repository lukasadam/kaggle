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
                   _log_transform_input_data,
                   _onehot_encode_input_data,
                   _impute_missing_data,
                   _exp1m_rmse)

from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (AdaBoostRegressor, 
                              RandomForestRegressor,
                              BaggingRegressor,
                              ExtraTreesRegressor,
                              GradientBoostingRegressor,
                              HistGradientBoostingRegressor)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from autogluon.tabular import TabularDataset, TabularPredictor

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error,
                             r2_score,
                             explained_variance_score)
from autogluon.core.metrics import make_scorer

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


class PricePredictionPipeline:
    """Pipeline for training and evaluating models for the price prediction task."""
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
        # Create the pipeline
        self.pipeline = Pipeline([
            ('load', FunctionTransformer(func=_load_input_data)),
            ('custom', FunctionTransformer(func=_custom_transform_data)),
            ('log_transform', FunctionTransformer(func=_log_transform_input_data)),
            # Add more steps like imputing, encoding, modeling...
        ])
        # Set the scaler
        self.scaler = StandardScaler()
        # Set the target variable
        self.target_var = target_var
        # Specify autogluon path
        self.autogluon_path = self.intermediate_dir / "autogluon_output" 
        # Specify custom evaluation metric
        self.eval_metric = make_scorer(
            name='exp1m_rmse',
            score_func=_exp1m_rmse,
            greater_is_better=True
        )
        # Initialize the predictor
        self.predictor = TabularPredictor(label=self.target_var, 
                                          eval_metric=self.eval_metric,
                                          problem_type="regression",
                                          path=self.autogluon_path)

    def load_train_test(
        self, 
        train_csv_path: str, 
        test_csv_path: str
    ):
        """Load the training and test data from CSV files."""
        self.train_df = self.pipeline.transform(train_csv_path)
        # Set all categorical features
        self.cat_features = self.train_df.select_dtypes(include=[object]).columns.tolist()
        # Remove the target variable from the categorical features
        if self.target_var in self.cat_features:
            self.cat_features.remove(self.target_var)
        # Set all numerical features
        self.num_features = self.train_df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove the target variable from the numerical features
        if self.target_var in self.num_features:
            self.num_features.remove(self.target_var)
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
            self.train_df = self.encode_one_hot(self.train_df, train=True)
            self.test_df = self.encode_one_hot(self.test_df, train=False)
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

    def encode_one_hot(self, df: pd.DataFrame, train=True):
        # Perform one-hot encoding on categorical columns
        df_onehot = pd.get_dummies(df, columns=self.cat_features, drop_first=False)

        if train:
            # When fitting on train set, store column names
            self.onehot_columns = df_onehot.columns
        else:
            # Align test columns with train columns
            for col in self.onehot_columns:
                if col not in df_onehot:
                    df_onehot[col] = False  # Add missing columns with 0s
            df_onehot = df_onehot[self.onehot_columns]
            # Remove target var column
            df_onehot.drop(columns=[self.target_var], inplace=True)
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
            self.x_train, self.y_train, test_size=0.1, random_state=42
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

    def train_models(self):
        """Train the models on the training data using AutoGluon."""
        # First we assemble the df using self.x_train & self.y_train
        train_data = self.x_train.copy()
        train_data[self.target_var] = self.y_train
        self.predictor.fit(train_data,
                           num_bag_folds = 10,
                           num_bag_sets = 2,
                           keep_only_best = True,
                           presets="best_quality",
                           time_limit=1800,)
        print(f"✅ Models trained and saved to: {self.autogluon_path}")
    
    def load_models(self):
        """Load a previously trained AutoGluon predictor."""
        self.predictor = TabularPredictor.load(self.autogluon_path)
        print(f"📂 Predictor loaded from: {self.autogluon_path}")

    def evaluate_model(self, model_name: str):
        """Evaluate the models on the validation data."""
        # First we assemble the df using self.x_val & self.y_val
        val_data = self.x_val.copy()
        val_data[self.target_var] = self.y_val

        # Evaluate the models
        results = self.predictor.evaluate(val_data, 
                                          silent=True, 
                                          model=model_name,
                                          extra_metrics=["r2", "mae", "mse"])
        return results

    def save_results_and_plot_best_model(self):
        """Save the results and plot the best model."""
        # First we assemble the df using self.x_val & self.y_val
        val_data = self.x_val.copy()
        val_data[self.target_var] = self.y_val

        # Get the results from the predictor
        results_df = self.predictor.leaderboard(val_data, silent=True, extra_metrics=["r2", "mae", "mse"])
    
        # Save CSV of metrics
        csv_path = self.tables_dir / f"model_metrics_{self._get_timestamp()}.csv"
        results_df.to_csv(csv_path)
        print(f"\n✅ Results saved to: {csv_path}")

        # Identify best model by lowest score
        self.best_model = results_df.iloc[results_df["score_test"].idxmax()]["model"]
        print(f"\n🏆 Best model: {self.best_model}")

        # Get the predictions from the best model
        predictions = self.predictor.predict(self.x_val, model=self.best_model)

        # Plot scatter plot of true vs predicted values
        # Fit regression line through scatter
        plt.figure(figsize=(10, 6))
        sns.regplot(x=self.y_val, y=predictions, scatter=True, color='red')
        plt.title(f"True vs Predicted Values - {self.best_model}")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.plot([self.y_val.min(), self.y_val.max()], [self.y_val.min(), self.y_val.max()], 'k--', lw=2)
        plt.xlim(self.y_val.min(), self.y_val.max())
        plt.ylim(self.y_val.min(), self.y_val.max())
        plt.grid()
        plt.tight_layout()
        plt.savefig(self.plot_dir / f"true_vs_predicted_{self.best_model}_{self._get_timestamp()}.png")

        # Save the predictions to a CSV file
        self.save_predictions(model_name=self.best_model, data_split="val")
    
    def _get_timestamp(self):
        return datetime.now().strftime("%d_%M%S")
    
    def save_predictions(self, model_name: str, data_split: str = "val"):
        """
        Make predictions and save them to a CSV file.

        Parameters:
        - model_name: str, name of the model to use for prediction
        - data_split: str, one of ["val", "test"]
        """
        assert data_split in ["val", "test"], "data_split must be 'val' or 'test'"

        if data_split == "val":
            X = self.x_val
            predictions = self.predictor.predict(X, model=model_name)
            df = pd.DataFrame({
                "true_transport": self.y_val.values,
                "predicted_transport": predictions,
            }, index=self.sample_ids)
            filename = f"true_vs_predicted_{model_name}_{self._get_timestamp()}.csv"

        else:  # test
            X = self.x_test
            predictions = self.predictor.predict(X, model=model_name)
            df = pd.DataFrame({
                "price": np.expm1(predictions)
            }, index=self.x_test.index)
            filename = "predictions_test_set.csv"

        csv_path = self.tables_dir / filename
        df.to_csv(csv_path, index=(data_split == "val"))
        print(f"📄 Predictions saved to: {csv_path}")
        return df

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
    prediction_pipeline = PricePredictionPipeline(
        intermediate_dir,
        tables_dir, 
        plot_dir,
        target_var="price"
    )
    prediction_pipeline.load_train_test(train_csv_path, test_csv_path)
    prediction_pipeline.prepare_data(
        one_hot_encode=True,
        normalize=False,
        strategy={
            "categorical": "simple",
            "numerical": "iterative"
        },
        feature_selection=False
    )
    prediction_pipeline.train_test_split()
    prediction_pipeline.train_models()
    prediction_pipeline.load_models()
    prediction_pipeline.save_results_and_plot_best_model()

    best_model = prediction_pipeline.best_model
    prediction_pipeline.save_predictions(model_name=best_model, data_split="test")
