import torch
import numpy as np
from config import conf
import pickle

# Train the module net
def train_dl(net, loader, opt, criterion, modelpath, epochs=50):
    for epoch in range(epochs):
        losses = list()
        for x_batch, y_batch in loader:
            if conf.GPU == 1 and conf.DEEP_LEARN == True:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
            net.train()
            if conf.MODEL.__name__ == 'HAN':
                net._init_h_state(last_batch_size=len(x_batch), init=-1)
            y_hat = net(x_batch)
            y_hat = y_hat.squeeze()
            loss = criterion(y_hat, y_batch)
            if loss != None:
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
        print(f'Epoch = {epoch + 1}, Loss = {np.mean(losses)}')

        # After each epoch, generate explanations
        if conf.GPU == 1 and conf.DEEP_LEARN == True:
            explanations = integrate_explainability(net, x_batch)
            print(f'Explainability for epoch {epoch + 1}:', explanations)

    # dump model
    torch.save(net.state_dict(), modelpath)

# Trains a machine learning model
def train_ml(net, dataset, modelpath):
    X, y = dataset
    net.fit(X, y)

    with open(modelpath, 'wb') as f:
        pickle.dump(net, f)

    # After training, generate explanations
    explanations = integrate_explainability(net, X)
    print('Explainability:', explanations)

# Evaluate model and print results


# Main training logic


"""
    Train the module net

    Params:
        - net (nn.Module): the network to train
        - loader (DataLoader): the data loader of the training examples
        - opt (optim.SGD or optim.ADAM): the optimiser to apply
        - criterion: the loss function to apply
        - modelpath (str): the path of the file where to save the model
        - epochs (int): the training epochs

"""
def train_dl(net, loader, opt, criterion, modelpath, epochs=50):
    for epoch in range(epochs):
        losses = list()

        for x_batch, y_batch in loader:
            # the dataset "lives" in the CPU, so do our mini-batches
            # therefore, we need to send those mini-batches to the
            # device where the model "lives"
            if conf.GPU == 1 and conf.DEEP_LEARN == True:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()
        
            net.train()
            if conf.MODEL.__name__ == 'HAN':
                net._init_h_state(last_batch_size=len(x_batch), init=-1)
            # Makes predictions
            y_hat = net(x_batch)
            y_hat = y_hat.squeeze()
            # Computes loss
            loss = criterion(y_hat, y_batch)

            if loss != None:
                # Computes gradients
                # Updates parameters and zeroes gradients
                opt.zero_grad()
                loss.backward()
                opt.step()

                losses.append(loss.item())

        print(f'Epoch = {epoch + 1}, Loss = {np.mean(losses)}')

    # dump model
    torch.save(net.state_dict(), modelpath)

"""
    Trains a machine learning model

    Params:
        - net: the model
        - dataset (tuple): the dataset contains the following:
            * X (np.ndarray): the data corresponding to the training instances
            * y (np.array): the training ground truth labels
"""
def train_ml(net, dataset, modelpath):
    X, y = dataset

    net.fit(X, y)

    with open(modelpath, 'wb') as f:
        pickle.dump(net, f)
