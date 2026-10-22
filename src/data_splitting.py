import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split

# Split the dataset (TensorDataset) into training and testing instances
def split_dl_data(dataset, batch_size=16, test_split=0.3, shuffle=True):
    data_size = len(dataset)
    indices = list(range(data_size))
    split = int(np.floor(test_split * data_size))
    if shuffle:
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, drop_last=True)
    return train_loader, test_loader

# Splits the dataset (X and y) into training and testing instances
def split_ml_data(data, test_size=0.3, shuffle=True):
    X, y = data
    return train_test_split(X, y, test_size=test_size, shuffle=shuffle)

# Function to get model explainability data split
def split_explainability_data(X, y, test_size=0.3, shuffle=True):
    X_train, X_test, y_train, y_test = split_ml_data((X, y), test_size, shuffle)
    return (X_train, X_test), (y_train, y_test)
