import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from autogluon.core.models import AbstractModel
from category_encoders import TargetEncoder


class SVRModel(AbstractModel):
    def _fit(self, X, y, **kwargs):
        self.pipeline = make_pipeline(
            TargetEncoder(
                cols=(
                    X.select_dtypes(include=["category", "object"]).columns.to_list()
                ),
                smoothing=10,
                min_samples_leaf=2,
            ),
            StandardScaler().set_output(transform="pandas"),
        )
        X_transformed = self.pipeline.fit_transform(X, y)
        self.model = LinearSVR(**self.params)
        self.model.fit(X_transformed, y.astype(float))

    def _predict_proba(self, X, **kwargs):
        X_transformed = self.pipeline.transform(X)
        return self.model.predict(X_transformed)

    def _predict(self, X, **kwargs):
        X_transformed = self.pipeline.transform(X)
        return self.model.predict(X_transformed)


