import torch

from torch import nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
        Creates a SimpleCNN object with the following layers:
            * conv2 (relu) -> pool2d -> conv2d (relu) -> pool2d 
            -> flatten -> fc (relu) -> fc (relu) -> fc (sigmoid)

        Params:
            - input_shape (tuple): the input shape (channels=1, time_len, features)
            - kernel_conv_size (int): the kernel size of the conv layers
            - kernel_pool_size (int): the kernel size of the pooling layers
            - padding_size (int): the padding size for the conv layers

        return object (SimpleCNN)
    """
    def __init__(self, input_shape, kernel_conv_size=5,\
        kernel_pool_size=2, padding_size=2):
        super(SimpleCNN, self).__init__()

        c, h, w = input_shape

        self.kernel_conv = 5
        self.kernel_pool = 2
        self.padding = 2

        self.conv1 = nn.Conv2d(c, 6, kernel_size=self.kernel_conv,\
            padding=self.padding)
        h, w = self.__calc_size(h,w)

        self.pool1 = nn.MaxPool2d(kernel_size=self.kernel_pool)
        h, w = self.__calc_size(h,w,conv_or_pool='pool')

        self.conv2 = nn.Conv2d(6, 16, kernel_size=self.kernel_conv,\
            padding=self.padding)
        h, w = self.__calc_size(h,w)

        self.pool2 = nn.MaxPool2d(kernel_size=self.kernel_pool)
        h, w = self.__calc_size(h,w,conv_or_pool='pool')

        self.fc1 = nn.Linear(16*h*w, 20) # calculated after flattening
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 1)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x)) # get the output probability

        return x


    def __calc_size(self, h, w, conv_or_pool='conv'):
        return (int(h + 2*self.padding - self.kernel_conv + 1),\
            int(w + 2*self.padding - self.kernel_conv + 1)) if conv_or_pool == 'conv'\
                else (int((h - self.kernel_pool)/self.kernel_pool + 1),\
                    int((w - self.kernel_pool)/self.kernel_pool + 1))