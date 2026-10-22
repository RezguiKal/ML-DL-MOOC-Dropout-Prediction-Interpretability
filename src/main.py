
import config.conf as conf
import os
import sys
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import ADASYN

from models.dl import *
from models.ml import *
from models.ensemble import kotsiantis, RF, AdaBoost, XGBoost

from reader import DataReader
from train_model import train
from test_model import test

import data_splitting as split
import utils


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Apply ADASYN only to the training data
def apply_adasyn(X_train, y_train):
    """
    Apply ADASYN to training data only.

    The input is flattened for resampling, then reshaped to its
    original per-instance dimensions.
    """
    original_shape = X_train.shape
    X_flat = X_train.reshape(X_train.shape[0], -1)

    oversampler = ADASYN(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(
        X_flat, y_train
    )

    X_resampled = X_resampled.reshape(
        X_resampled.shape[0], *original_shape[1:]
    )

    return X_resampled, y_resampled


# Rebuild the training loader after oversampling
def oversample_dl_train_loader(train_loader):
    """
    Apply ADASYN to the training TensorDataset only.

    This assumes split_dl_data returns a DataLoader whose dataset
    is a TensorDataset containing features and labels.
    """
    train_dataset = train_loader.dataset

    if not isinstance(train_dataset, TensorDataset):
        raise TypeError(
            "ADASYN for deep learning requires the training loader "
            "to use a TensorDataset."
        )

    X_train = train_dataset.tensors[0].cpu().numpy()
    y_train = train_dataset.tensors[1].cpu().numpy().reshape(-1)

    X_resampled, y_resampled = apply_adasyn(X_train, y_train)

    X_tensor = torch.as_tensor(
        X_resampled, dtype=train_dataset.tensors[0].dtype
    )
    y_tensor = torch.as_tensor(
        y_resampled, dtype=train_dataset.tensors[1].dtype
    )

    resampled_dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        resampled_dataset,
        batch_size=train_loader.batch_size or conf.BATCH_SIZE,
        shuffle=True,
        num_workers=getattr(train_loader, 'num_workers', 0),
        pin_memory=getattr(train_loader, 'pin_memory', False)
    )


if __name__ == '__main__':

    set_seed(42)

    # The dataset name to use
    if len(sys.argv) < 2:
        raise ValueError(
            "Please provide a dataset name. "
            "Example: python main.py KDDCup"
        )

    dataset = sys.argv[1]
    dataset_path = os.path.join(conf.DATA_DIR, dataset)

    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_path}"
        )

    # Identify the selected model
    model_name = conf.MODEL.__name__

    # Models that expect four-dimensional CNN-style input
    # Keep this list restricted to architectures that require it.
    cnn_4d_models = {'SimpleCNN', 'CFIN', 'ConRec'}

    # Models that need sequential input: (samples, timesteps, features)
    sequence_models = {
        'SimpleLSTM',
        'LSTM_ATT',
        'CNN_LSTM',
        'CNN_LSTM_ATT',
        'HAN'
    }

    # Models that explicitly expect flattened feature vectors
    flattened_models = {
        'LogisticRegressor',
        'DTree',
        'SVM',
        'NaiveBayes',
        'KNN',
        'LDA',
        'WINNOW',
        'MajorityClassBaseline',
        'RF',
        'AdaBoost',
        'XGBoost',
        'KotsiantisEnsemble',
        'MLayeredDNN'
    }

    # Select device for deep learning
    use_gpu = (
        getattr(conf, 'GPU', 0) == 1
        and torch.cuda.is_available()
        and conf.DEEP_LEARN
    )

    device = torch.device('cuda' if use_gpu else 'cpu')
    print(f'Selected model: {model_name}')
    print(f'Deep learning: {conf.DEEP_LEARN}')
    print(f'Device: {device}')

    for time_module in sorted(os.listdir(dataset_path)):

        time_module_path = os.path.join(dataset_path, time_module)

        # Skip non-directories
        if not os.path.isdir(time_module_path):
            continue

        csv_path = os.path.join(
            time_module_path, 'timeseries.csv'
        )

        # Skip directories without the expected dataset file
        if not os.path.isfile(csv_path):
            print(f'Skipping {time_module}: {csv_path} not found.')
            continue

        print(f'\nWorking for time module {time_module}')

        reader = DataReader(csv_path, lag=conf.LAG)
        X, y = reader.read_data()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).reshape(-1)

        if len(X) != len(y):
            raise ValueError(
                f"X and y have different sample counts: "
                f"{len(X)} and {len(y)}"
            )

        # Reshape instances according to the selected model
        # Keep sequential input for temporal models.
        if model_name in cnn_4d_models:
            if X.ndim == 3:
                X = X.reshape(
                    X.shape[0], 1, X.shape[1], X.shape[2]
                )

        elif model_name in flattened_models or not conf.DEEP_LEARN:
            X = X.reshape(X.shape[0], -1)

        elif model_name in sequence_models:
            if X.ndim != 3:
                raise ValueError(
                    f"{model_name} expects input with shape "
                    "(samples, timesteps, features), but received "
                    f"{X.shape}."
                )

        print(
            f'Input shape: {X.shape}\t'
            f'Labels shape: {y.shape}'
        )

        # Create an empty dataframe to contain the testing performances
        performances = pd.DataFrame(
            columns=['y_true', 'y_pred']
        )

        for run in range(conf.RUNS):

            print(f'\nWorking with {model_name} @run {run + 1}')

            # Create the model
            model = utils.create_model(X)

            if model is None:
                raise ValueError(
                    f"Could not create model: {model_name}. "
                    "Check the model factory in utils.py."
                )

            if conf.DEEP_LEARN:
                model = model.to(device)

            # Reset the weights for precautionary measures.
            # This prevents carrying learned weights across runs.
            utils.weight_reset(model)

            # Create the model dump path
            models_path = utils.create_model_dir(
                dataset, time_module, model_name, str(run)
            )

            if run == 0:
                print(f'Model summary:\n{model}')

            if conf.DEEP_LEARN:

                # Build the dataset for deep learning
                data = TensorDataset(
                    torch.from_numpy(X).float(),
                    torch.from_numpy(y).float()
                )

                train_loader, test_loader = split.split_dl_data(
                    data, batch_size=conf.BATCH_SIZE
                )

                # Oversample only the training data to avoid leakage
                if getattr(conf, 'OVERSAMPLING', False):
                    train_loader = oversample_dl_train_loader(
                        train_loader
                    )

                # Configure the optimizer
                # Use the existing optimizer and hyperparameters
                if model_name == 'HAN':
                    opt = conf.OPTIMISER(
                        filter(
                            lambda p: p.requires_grad,
                            model.parameters()
                        ),
                        lr=conf.LR,
                        momentum=conf.MOMENTUM
                    )
                else:
                    opt = conf.OPTIMISER(
                        model.parameters(),
                        lr=conf.LR,
                        weight_decay=conf.WEIGHT_DECAY
                    )

                # Train the deep learning model
                train.train_dl(
                    model,
                    train_loader,
                    opt=opt,
                    criterion=conf.LOSS_CRITERION,
                    modelpath=os.path.join(
                        models_path, 'model'
                    ),
                    epochs=conf.EPOCHS
                )

                # Evaluate the deep learning model
                performances = test.test_dl(
                    model,
                    test_loader,
                    performances
                )

            else:

                # Split the data before applying oversampling
                X_train, X_test, y_train, y_test = (
                    split.split_ml_data((X, y))
                )

                # Oversample training data only
                if getattr(conf, 'OVERSAMPLING', False):
                    X_train, y_train = apply_adasyn(
                        X_train, y_train
                    )

                # Train the machine learning or ensemble model
                train.train_ml(
                    model,
                    dataset=(X_train, y_train),
                    modelpath=os.path.join(
                        models_path, 'model'
                    )
                )

                # Evaluate the model on the untouched test set
                performances = test.test_ml(
                    model,
                    dataset=(X_test, y_test),
                    performances=performances
                )

        # Save the testing performances
        scores_dir_path = utils.create_scores_dir(
            dataset, time_module, model_name
        )

        scores_file = os.path.join(
            scores_dir_path, 'scores.csv'
        )

        performances.to_csv(scores_file, index=False)

        print(f'Saved performance results to {scores_file}')

