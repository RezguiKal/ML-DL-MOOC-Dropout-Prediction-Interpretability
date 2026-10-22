
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import mode

import config.conf as conf

# Import machine learning models
from models.ml import baselines

# Import deep learning models
from models.dl import (
    simple_cnn,
    simple_lstm,
    lstm_att,
    cnn_lstm,
    cnn_lstm_att,
    dnns,
    cfin,
    conrec,
    han
)

# Import ensemble learning models
from models.ensemble import kotsiantis, RF, AdaBoost, XGBoost


# Creates the model dump directory if it doesn't exist
def create_model_dir(dataset, time_module, model_name, run):
    algo_dir_path = os.path.join(
        conf.MODEL_DIR,
        dataset,
        time_module,
        str(conf.LAG) + 'days',
        model_name,
        'models',
        'run',
        str(run)
    )

    Path(algo_dir_path).mkdir(parents=True, exist_ok=True)
    print(f'Directory {algo_dir_path} created.')
    return algo_dir_path


# Creates the scores dump directory if it doesn't exist
def create_scores_dir(dataset, time_module, model_name):
    scores_dir_path = os.path.join(
        conf.PERFORMANCE_DIR,
        dataset,
        time_module,
        str(conf.LAG) + 'days',
        model_name
    )

    Path(scores_dir_path).mkdir(parents=True, exist_ok=True)
    print(f'Directory {scores_dir_path} created.')
    return scores_dir_path


# Get the parameters associated with the selected model
def __get_model_parameters():
    return conf.PARAMETERS.get(conf.MODEL.__name__, {})


# Deep learning model constructors

def __simple_cnn(X):
    return simple_cnn.SimpleCNN(
        X.shape[1:],
        **__get_model_parameters()
    )


def __simple_lstm(X):
    return simple_lstm.SimpleLSTM(
        X.shape[1:],
        **__get_model_parameters()
    )


def __lstm_att(X):
    return lstm_att.LSTM_ATT(
        X.shape[1:],
        **__get_model_parameters()
    )


def __cnn_lstm(X):
    return cnn_lstm.CNN_LSTM(
        X.shape[1:],
        **__get_model_parameters()
    )


def __cnn_lstm_att(X):
    return cnn_lstm_att.CNN_LSTM_ATT(
        X.shape[1:],
        **__get_model_parameters()
    )


def __layered_dnn(X):
    return dnns.MLayeredDNN(
        X.shape,
        **__get_model_parameters()
    )


def __cfin(X):
    return cfin.CFIN(
        X,
        **__get_model_parameters()
    )


def __conrec(X):
    return conrec.ConRec(
        X.shape[1:],
        **__get_model_parameters()
    )


def __hierarchical_attention_net(X):
    return han.HAN(
        X,
        **__get_model_parameters()
    )


# Ensemble learning model constructors

def __kotsiantis_ensemble(X):
    return kotsiantis.KotsiantisEnsemble(
        **__get_model_parameters()
    )


def __rf(X):
    return RF(
        **__get_model_parameters()
    )


def __adaboost(X):
    return AdaBoost(
        **__get_model_parameters()
    )


def __xgboost(X):
    return XGBoost(
        **__get_model_parameters()
    )


# Traditional machine learning model constructors

def __logistic_regressor(X):
    return baselines.LogisticRegressor(
        **__get_model_parameters()
    )


def __naive_bayes(X):
    return baselines.NaiveBayes(
        **__get_model_parameters()
    )


def __svm(X):
    return baselines.SVM(
        **__get_model_parameters()
    )


def __dtree(X):
    return baselines.DTree(
        **__get_model_parameters()
    )


def __knn(X):
    return baselines.KNN(
        **__get_model_parameters()
    )


def __lda(X):
    return baselines.LDA(
        **__get_model_parameters()
    )


def __winnow(X):
    return baselines.WINNOW(
        **__get_model_parameters()
    )


def __majority_class(X):
    return baselines.MajorityClassBaseline(
        **__get_model_parameters()
    )


# Creates the model based on the model selected in conf.py
def create_model(X):

    # Map model class names to their constructor functions
    switch = {
        # Deep learning models
        simple_cnn.SimpleCNN.__name__: __simple_cnn,
        simple_lstm.SimpleLSTM.__name__: __simple_lstm,
        lstm_att.LSTM_ATT.__name__: __lstm_att,
        cnn_lstm.CNN_LSTM.__name__: __cnn_lstm,
        cnn_lstm_att.CNN_LSTM_ATT.__name__: __cnn_lstm_att,
        dnns.MLayeredDNN.__name__: __layered_dnn,
        cfin.CFIN.__name__: __cfin,
        conrec.ConRec.__name__: __conrec,
        han.HAN.__name__: __hierarchical_attention_net,

        # Machine learning models
        baselines.LogisticRegressor.__name__: __logistic_regressor,
        baselines.NaiveBayes.__name__: __naive_bayes,
        baselines.SVM.__name__: __svm,
        baselines.DTree.__name__: __dtree,
        baselines.KNN.__name__: __knn,
        baselines.LDA.__name__: __lda,
        baselines.WINNOW.__name__: __winnow,
        baselines.MajorityClassBaseline.__name__: __majority_class,

        # Ensemble learning models
        RF.__name__: __rf,
        AdaBoost.__name__: __adaboost,
        XGBoost.__name__: __xgboost,
        kotsiantis.KotsiantisEnsemble.__name__: __kotsiantis_ensemble
    }

    model_name = conf.MODEL.__name__
    constructor = switch.get(model_name)

    if constructor is None:
        available_models = ', '.join(sorted(switch.keys()))
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Available models: {available_models}"
        )

    return constructor(X)


# Reset model weights before retraining a deep learning model
def weight_reset(m):
    if isinstance(
        m,
        (nn.Conv1d, nn.Conv2d, nn.Linear, nn.Embedding, nn.LSTM)
    ):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()


# Function for feature importance analysis
def feature_importance(X, model):
    """
    Return native feature importance or coefficients when available.

    Note:
    - Tree-based models often expose feature_importances_.
    - Linear models often expose coef_.
    - Deep learning models generally require a separate explanation method.
    """

    if hasattr(model, 'feature_importances_'):
        return model.feature_importances_

    elif hasattr(model, 'coef_'):
        return np.abs(model.coef_)

    else:
        raise ValueError(
            f"{model.__class__.__name__} does not expose native "
            "feature importance. Use SHAP, LIME, or a suitable "
            "deep-learning explanation method."
        )


# Function to generate explanations using SHAP
def explain_with_shap(model, X):
    """
    Generate SHAP explanations.

    For sklearn-compatible models, SHAP can often infer an appropriate
    explainer. For PyTorch models, provide a suitable prediction function
    or use a model-specific SHAP explainer.
    """
    import shap

    # Convert tensors to NumPy when necessary
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    # Use a general explainer for supported model types
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    return shap_values


# Function to create explainability visualizations
def plot_explanations(explanation, plot_type='bar'):
    """
    Plot SHAP explanations.

    Supported plot types:
    - bar: global feature importance
    - summary: beeswarm summary of feature effects
    """
    import shap

    if plot_type == 'bar':
        shap.plots.bar(explanation)

    elif plot_type == 'summary':
        shap.plots.beeswarm(explanation)

    else:
        raise ValueError(
            "Invalid plot type. Choose 'bar' or 'summary'."
        )


# Function for creating Grad-CAM explanations for CNNs
def explain_cnn_with_gradcam(model, X, target_layer=None):
    """
    Generate Grad-CAM explanations when a compatible convolutional
    target layer and GradCAM implementation are available.

    The target layer must be supplied explicitly because different
    CNN architectures do not necessarily have a layer named 'layer4'.
    """
    import torch
    from utils.gradcam import GradCAM

    if target_layer is None:
        raise ValueError(
            "Please provide target_layer for the selected CNN architecture."
        )

    model.eval()

    if isinstance(X, np.ndarray):
        X = torch.as_tensor(X, dtype=torch.float32)

    cam = GradCAM(model, target_layer)
    return cam(X)


# Function to interpret ensemble model predictions
def ensemble_interpretation(models, X):
    """
    Combine predictions using majority voting.

    This function assumes all models implement predict(X) and return
    compatible class labels.
    """
    if not models:
        raise ValueError("At least one model is required.")

    predictions = np.asarray([
        np.asarray(model.predict(X)).reshape(-1)
        for model in models
    ])

    # Shape: number of models x number of samples
    return mode(predictions, axis=0, keepdims=False).mode


# Function to integrate different explainability methods
def integrate_explainability(model, X):
    """
    Attempt native feature importance and SHAP explanations.

    Grad-CAM is not run automatically because the correct target layer
    depends on the specific CNN architecture.
    """
    explanations = {
        'feature_importance': None,
        'shap': None,
        'grad_cam': None
    }

    try:
        explanations['feature_importance'] = feature_importance(X, model)
    except (ValueError, AttributeError):
        pass

    try:
        explanations['shap'] = explain_with_shap(model, X)
    except (ValueError, TypeError, AttributeError, ImportError):
        pass

    return explanations
