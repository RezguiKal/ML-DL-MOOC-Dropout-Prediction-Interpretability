
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from config import conf


class BaselineClassifier:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, np.asarray(y).ravel())
        return self

    def predict(self, X):
        return np.asarray(self.model.predict(X)).ravel()

    def feature_importance(self, X=None):
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_)
        else:
            raise ValueError(
                f"{type(self.model).__name__} does not expose "
                "built-in feature importance or coefficients."
            )

    def explain_with_shap(self, X):
        import shap

        explainer = shap.Explainer(self.model, X)
        return explainer(X)


def _get_parameters(class_name):
    """Retrieve model parameters safely from the configuration."""
    return conf.PARAMETERS.get(class_name) or {}


class SVM(BaselineClassifier):
    def __init__(self):
        super().__init__(SVC(**_get_parameters("SVM")))


class LogisticRegressor(BaselineClassifier):
    def __init__(self):
        super().__init__(
            LogisticRegressionCV(**_get_parameters("LogisticRegressor"))
        )


class DTree(BaselineClassifier):
    def __init__(self):
        super().__init__(
            DecisionTreeClassifier(**_get_parameters("DTree"))
        )


class NaiveBayes(BaselineClassifier):
    def __init__(self):
        super().__init__(GaussianNB(**_get_parameters("NaiveBayes")))


class KNN(BaselineClassifier):
    def __init__(self):
        super().__init__(
            KNeighborsClassifier(**_get_parameters("KNN"))
        )


class LDA(BaselineClassifier):
    """Linear Discriminant Analysis baseline."""

    def __init__(self):
        super().__init__(
            LinearDiscriminantAnalysis(**_get_parameters("LDA"))
        )


class MajorityClassBaseline(BaselineClassifier):
    def __init__(self):
        super().__init__(None)

    def fit(self, X, y):
        y = np.asarray(y).ravel()
        classes, counts = np.unique(y, return_counts=True)
        self.majority_cls = classes[np.argmax(counts)]
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.majority_cls)


class WINNOW(BaselineClassifier):
    def __init__(self, lr=1e-3, bias=5e-2, alpha=2, epochs=100):
        super().__init__(None)
        if alpha <= 1:
            raise ValueError("alpha must be greater than 1.")
        if epochs < 1:
            raise ValueError("epochs must be positive.")

        self.lr = lr
        self.bias = bias
        self.alpha = alpha
        self.epochs = epochs

    def __update_weights(self, X, y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()

        false_negative = np.where(
            (y_true != y_pred) & (y_true == 1)
        )[0]
        false_positive = np.where(
            (y_true != y_pred) & (y_true == 0)
        )[0]

        rows, cols = np.where(X != 0)

        increase = np.unique(
            cols[np.isin(rows, false_negative)]
        )
        decrease = np.unique(
            cols[np.isin(rows, false_positive)]
        )

        self.weights[increase] *= self.alpha
        self.weights[decrease] = 0

    def fit(self, X, y):
        y = np.asarray(y).ravel()
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("WINNOW requires binary labels encoded as 0 and 1.")

        self.weights = np.ones(X.shape[1]) + self.bias
        self.theta = X.shape[1] / 2

        for _ in range(self.epochs):
            y_pred = self.predict(X)
            self.__update_weights(X, y, y_pred)

        return self

    def predict(self, X):
        scores = np.asarray(X @ self.weights).ravel()
        return (scores >= self.theta).astype(int)

    def __str__(self):
        return (
            f"WINNOW(lr={self.lr}, bias={self.bias}, "
            f"alpha={self.alpha}, epochs={self.epochs})"
        )
