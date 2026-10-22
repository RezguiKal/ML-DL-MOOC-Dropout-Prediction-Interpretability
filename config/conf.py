
from os.path import join
from os import makedirs

from torch import nn, optim

from models.dl import (
    simple_cnn,
    simple_lstm,
    lstm_att,
    cnn_lstm,
    cnn_lstm_att,
    dnns,
    cfin,
    conrec,
    han,
)
from models.ml import baselines
from models.ensemble import kotsiantis, RF, AdaBoost, XGBoost


def create_dirs(dirname):
    """Create a directory and its parent directories."""
    try:
        makedirs(dirname, exist_ok=True)
        print(f"Directory ready: {dirname}")
    except OSError as exc:
        print(f"Could not create directory {dirname}: {exc}")
        raise


# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------
DATA_DIR = join("..", "resources", "data")
MODEL_DIR = join("..", "resources", "models")
PERFORMANCE_DIR = join("..", "resources", "performances")

for directory in (DATA_DIR, MODEL_DIR, PERFORMANCE_DIR):
    create_dirs(directory)


# ---------------------------------------------------------
# General training configuration
# ---------------------------------------------------------
GPU = 0                  # Set to 1 to enable GPU, if supported
LR = 1e-3
#LR = 0.1 # for HAN
#MOMENTUM = 0.9 # for HAN
WEIGHT_DECAY = 1e-5
OPTIMISER = optim.Adam # for any other method
#OPTIMISER = optim.SGD # for HAN
BATCH_SIZE = 512
OVERSAMPLING = True

LOSS_CRITERION = nn.BCELoss()
# how much to get from the timeseries prior to prediction
LAG = 7
RUNS = 1 # how many times to perform the training
EPOCHS = 100


# ---------------------------------------------------------
# Select the model to train
# ---------------------------------------------------------
DEEP_LEARN = False

# Traditional ML
MODEL = baselines.LogisticRegressor
# MODEL = baselines.MajorityClassBaseline
# MODEL = baselines.DTree
# MODEL = baselines.SVM
# MODEL = baselines.NaiveBayes
# MODEL = baselines.KNN
# MODEL = baselines.LDA
# MODEL = baselines.WINNOW

# Ensemble models
# MODEL = RF
# MODEL = AdaBoost
# MODEL = XGBoost
# MODEL = kotsiantis.KotsiantisEnsemble

# Deep learning
# DEEP_LEARN = True
# MODEL = simple_cnn.SimpleCNN
# MODEL = simple_lstm.SimpleLSTM
# MODEL = lstm_att.LSTM_ATT
# MODEL = cnn_lstm.CNN_LSTM
# MODEL = cnn_lstm_att.CNN_LSTM_ATT
# MODEL = dnns.MLayeredDNN
# MODEL = cfin.CFIN
# MODEL = conrec.ConRec
# MODEL = han.HAN


# ---------------------------------------------------------
# Default model hyperparameters
# Keys must match the class names used by each model.
# ---------------------------------------------------------
PARAMETERS = {

    # ================= Traditional ML =================

    baselines.LogisticRegressor.__name__: {
        "Cs": 10,
        "penalty": "l2",
        "max_iter": 15000,
        "cv": 3,
        "random_state": 42,
    },

    baselines.DTree.__name__: {
        "criterion": "gini",
        "random_state": 42,
    },

    baselines.SVM.__name__: {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True,
        "random_state": 42,
    },

    baselines.NaiveBayes.__name__: {
        "var_smoothing": 1e-9,
        "priors": None,
    },

    baselines.KNN.__name__: {
        "n_neighbors": 1, # for the Kotsiantis Ensemble this needs to be set to 1
        # for the other baselines to 2
        "n_jobs": -1,
    },

    baselines.LDA.__name__: {
        "solver": "svd",
    },

    baselines.WINNOW.__name__: {
        "lr": 1e-3,
        "bias": 5e-2,
        "alpha": 2,
        "epochs": 100,
    },

    # ================= Ensemble models =================

    RF.__name__: {
        "n_estimators": 100,
        "criterion": "gini",
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
    },

    AdaBoost.__name__: {
        "n_estimators": 100,
        "learning_rate": 1.0,
        "random_state": 42,
    },

    XGBoost.__name__: {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "logloss",
    },

    kotsiantis.KotsiantisEnsemble.__name__: {},

    # ================= Simple CNN =================

    simple_cnn.SimpleCNN.__name__: {
        "kernel_conv_size": 5,
        "kernel_pool_size": 2,
        "padding_size": 2,
    },

    # ================= Simple LSTM =================

    simple_lstm.SimpleLSTM.__name__: {
        "h_size": 100,
    },

    # ================= LSTM with attention =================

    lstm_att.LSTM_ATT.__name__: {
        "h_size": 100,
    },

    # ================= CNN-LSTM =================

    cnn_lstm.CNN_LSTM.__name__: {
        "h_size": 100,
    },

    # ================= CNN-LSTM-Attention =================

    cnn_lstm_att.CNN_LSTM_ATT.__name__: {
        "lstm_hidden_size": 256,
        "num_heads": 16,
        "attention_dropout": 0.2,
        "fc_dropout": 0.5,
        "positional_base": 1000,
    },

    # ================= DNN =================

    dnns.MLayeredDNN.__name__: {
        "m": 5,
        "alpha": 2,
    },

    # ================= CFIN =================

    cfin.CFIN.__name__: {
        "f_embed_size": 100,
        "ctx_size": 100,
        "ctx_embed_size": 50,
        "attn_size": 25,
    },

    # ================= ConRec =================

    conrec.ConRec.__name__: {
        "h_size": 50,
        "kernel_conv_size": 2,
        "kernel_pool_size": 2,
        "num_channels": [20, 50],
    },

    # ================= HAN =================

    han.HAN.__name__: {
        "embed_size": 5,
        "lcl_h_size": 50,
        "glb_h_size": 50,
    },
}
