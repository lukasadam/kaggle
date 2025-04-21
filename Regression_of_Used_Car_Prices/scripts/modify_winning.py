from tabnanny import verbose
import numpy as np
import pandas as pd
import datetime
import shap
import random
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder,
)
from sklearn.linear_model import Ridge

from category_encoders import TargetEncoder

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.svm import LinearSVR, SVR
from autogluon.tabular import TabularPredictor

# from xgboost import XGBRegressor

import warnings

warnings.filterwarnings("ignore")

from utils import _bin_price, _custom_transform_data

seed = 42
VER = 8
n_folds = 3
n_bags = 3
TESTRUN = False

df_train = pd.read_csv(
    "/Users/adaml9/Private/kaggle/Regression_of_Used_Car_Prices/data/playground-series-s4e9/train.csv",
    index_col=0,
)
df_test = pd.read_csv(
    "/Users/adaml9/Private/kaggle/Regression_of_Used_Car_Prices/data/playground-series-s4e9/test.csv",
    index_col=0,
)
sample_sub = pd.read_csv(
    "/Users/adaml9/Private/kaggle/Regression_of_Used_Car_Prices/data/playground-series-s4e9/sample_submission.csv"
)

# prepare dataframes for the oof and test predictions:
oof_df = df_train.reset_index()[["id"]]
test_df = df_test.reset_index()[["id"]]

if TESTRUN:
    df_train = df_train.sample(frac=0.2, random_state=seed)
    oof_df = oof_df.sample(frac=0.2, random_state=seed)

feat_cols = df_test.columns.to_list()
data_concat = _custom_transform_data(pd.concat([df_train[feat_cols], df_test]))
data_train = data_concat.loc[df_train.index]
# Merge price column back to the training data
data_train["price"] = df_train["price"]
data_test = data_concat.loc[df_test.index]

# Get all categorical columns from dataframe
cat_cols = data_train.select_dtypes(include=["object"]).columns.to_list()
for col in cat_cols:
    # Convert column to categorical type
    data_train[col] = data_train[col].astype("category")
    data_test[col] = data_test[col].astype("category")

# Get all numerical columns from dataframe
num_cols = data_train.select_dtypes(include=["number"]).columns.to_list()

data_train2 = _bin_price(data_train)

oof_df = pd.read_csv(f"oof_v{VER}.csv", index_col=0)
oof_df["price"] = df_train["price"]
oof_df = oof_df.reset_index()
oof_df = oof_df.drop(columns="id")
test_df = pd.read_csv(f"test_v{VER}.csv")
test_df = test_df.set_index("id")

predictor = TabularPredictor(
    label="price", problem_type="regression", eval_metric="rmse"
)
predictor.fit(
    train_data=oof_df,
    included_model_types=["GBM"],
    num_bag_folds=2,  # disable bagging
    num_stack_levels=1,  # disable stacking
    time_limit=180,
    keep_only_best=True,  # keep only the best model
    presets=None,  # don't use a preset that would add ensembling
)
results_df = predictor.leaderboard(oof_df, silent=False)
scores = results_df[["model", "score_val"]].set_index("model")["score_val"].to_dict()
scores["SVR"] = score_svr
scores["LGBM1"] = score_lgb1
scores["LGBM1_st"] = score_lgb1_st
scores["LGBM5_st"] = score_lgb5_st
scores = pd.Series(scores).to_frame(name="score")
scores["score"] = -1 * scores["score"]
scores = scores.sort_values(by="score", ascending=False)

plt.figure(figsize=(10, 6))
barplot = sns.barplot(
    data=scores,
    x="score",
    y=scores.index,
    palette=["grey", "#a3a3a3", "#d3d3d3", "#545454", "#bdbdbd", "orange"],
)
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.savefig("models_scores.png", dpi=300)

# Predict final result
predictions = predictor.predict(
    test_df,
    as_pandas=True,
)
df = pd.DataFrame({
    "price": predictions.values
}, index=test_df.index)
df.to_csv("predictions_test_set.csv")

