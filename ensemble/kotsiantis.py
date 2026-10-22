
import numpy as np
from scipy.stats import mode

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from config import conf
from models.ml.baselines import (
    BaselineClassifier,
    NaiveBayes,
    KNN,
    WINNOW,
)


def get_parameters(model_name):
    return conf.PARAMETERS.get(model_name) or {}


class RF(BaselineClassifier):
    """Random Forest ensemble classifier."""

    def __init__(self):
        super().__init__(
            RandomForestClassifier(**get_parameters("RF"))
        )


class AdaBoost(BaselineClassifier):
    """AdaBoost ensemble classifier."""

    def __init__(self):
        super().__init__(
            AdaBoostClassifier(**get_parameters("AdaBoost"))
        )


class XGBoost(BaselineClassifier):
    """XGBoost gradient-boosted tree classifier."""

    def __init__(self):
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "XGBoost is required. Install it using: pip install xgboost"
            ) from exc

        super().__init__(
            XGBClassifier(**get_parameters("XGBoost"))
        )


class KotsiantisEnsemble(BaselineClassifier):
    """
    Hard-voting ensemble combining the original classifiers
    with optional tree-based ensemble models.
    """

    def __init__(self, include_tree_ensembles=True):
        self.components = [
            WINNOW(),
            NaiveBayes(),
            KNN(),
        ]

        if include_tree_ensembles:
            self.components.extend([
                RF(),
                AdaBoost(),
                XGBoost(),
            ])

        super().__init__(None)

    def fit(self, X, y):
        y = np.asarray(y).ravel()

        for comp in self.components:
            comp.fit(X, y)

        return self

    def predict(self, X):
        predictions = np.column_stack([
            np.asarray(comp.predict(X)).ravel()
            for comp in self.components
        ])

        # Majority vote across classifiers for each sample.
        # In a tie, scipy's mode returns the smallest label
        # for numeric class labels.
        return mode(
            predictions, axis=1, keepdims=False
        ).mode

    def ensemble_interpretation(self, X):
        """Return individual predictions and majority-vote predictions."""
        individual_predictions = np.column_stack([
            np.asarray(comp.predict(X)).ravel()
            for comp in self.components
        ])

        majority_vote = mode(
            individual_predictions,
            axis=1,
            keepdims=False,
        ).mode

        return {
            "individual_predictions": individual_predictions,
            "majority_vote": majority_vote,
        }

    def integrate_explainability(self, X):
        """Generate SHAP explanations for each component."""
        explanations = {}

        for comp in self.components:
            name = type(comp).__name__

            try:
                explanations[name] = comp.explain_with_shap(X)
            except Exception as exc:
                # Some explainers require specific input formats,
                # background data, or model-specific configuration.
                explanations[name] = {
                    "error": str(exc)
                }

        return explanations
