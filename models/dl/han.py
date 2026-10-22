import numpy as np

import torch
import torch.nn as nn

import config.conf as conf

import torch.nn.functional as F


def matrix_mul(input, weight, bias=False):
    feature_list = []

    for feature in input:
        #assert not torch.isnan(feature).any()
        feature = feature.mm(weight)
        #assert not torch.isnan(feature).any()
        if isinstance(bias, torch.nn.parameter.Parameter):
            feature = feature + bias.expand(feature.size()[0], bias.size()[1])
        #assert not torch.isnan(feature).any()

        feature = torch.tanh(feature).unsqueeze(0)
        #assert not torch.isnan(feature).any()

        feature_list.append(feature)

    return torch.cat(feature_list, dim=0).squeeze()

def element_wise_mul(input1, input2):

    feature_list = []
    for feature_1, feature_2 in zip(input1, input2):
        feature_2 = feature_2.unsqueeze(1).expand_as(feature_1)
        feature = feature_1 * feature_2
        feature_list.append(feature.unsqueeze(0))
    
    output = torch.cat(feature_list,0)

    return torch.sum(output, dim=1).unsqueeze(1)

class LocalAttNet(nn.Module):
    def __init__(self, vec_size, embed_num=100,\
        embed_size=100, h_size=50):
        super(LocalAttNet, self).__init__()

        embed_num = int(embed_num + 1)

        self.weight = nn.Parameter(torch.Tensor(2 * h_size, 2 * h_size)) 
        self.bias = nn.Parameter(torch.Tensor(1, 2 * h_size))
        self.ctx_weight = nn.Parameter(torch.Tensor(2 * h_size, 1))

        self.lookup = nn.Embedding(embed_num, embed_size)
        
        self.gru = nn.GRU(embed_size, h_size,\
            bidirectional=True, batch_first=True)

        self._create_weights()

    def _create_weights(self, mean=0.0, std=0.05):
        self.weight.data.normal_(mean, std)
        self.bias.data.normal_(mean, std)
        self.ctx_weight.data.normal_(mean, std)

    def forward(self, input, h_state):
        output = self.lookup(input.long())
        # output.shape = (batch_size, vec_size, embed_size)
        f_output, h_output = self.gru(output.float(), h_state)
        # f_output.shape = (batch_size, vec_size, 2*h_size)
        # h_output.shape = (2, batch_size, 2*h_size)
        assert not torch.isnan(f_output).any()
        # attention mechanism
        output = matrix_mul(f_output, self.weight, self.bias)
        assert not torch.isnan(output).any()
        # output.shape = (batch_size, vec_size, 2*h_size)
        output = matrix_mul(output, self.ctx_weight)
        # output.shape = (batch_size, vec_size)
        output = F.softmax(output, dim=0)
        output = element_wise_mul(f_output, output)
        # output.shape = (batch_size, 1, 2*h_size)
        return output, h_output

class GlobalAttNet(nn.Module):

    def __init__(self, glb_h_size=50, lcl_h_size=50):
        super(GlobalAttNet, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(2 * glb_h_size, 2 * glb_h_size))
        self.bias = nn.Parameter(torch.Tensor(1, 2 * glb_h_size))
        self.ctx_weight = nn.Parameter(torch.Tensor(2 * glb_h_size, 1))

        self.gru = nn.GRU(2 * lcl_h_size, glb_h_size, bidirectional=True,\
                        batch_first=True)
        
        self.fc = nn.Linear(2 * glb_h_size, 1)

        self._create_weights()

    def _create_weights(self, mean=0.0, std=0.05):
        self.weight.data.normal_(mean, std)
        self.ctx_weight.data.normal_(mean, std)
        self.bias.data.normal_(mean, std)


    def forward(self, input, h_state):
        f_output, h_output = self.gru(input, h_state)
        # f_output.shape = (batch_size, LAG, 2 * glb_h_size)
        # h_output.shape = (2, batch_size, glb_h_size)

        output = matrix_mul(f_output, self.weight, self.bias)
        # output.shape = (batch_size, LAG, 2 * glb_h_size)
        output = matrix_mul(output, self.ctx_weight).permute(1,0)
        # output.shape = (LAG, batch_size)
        output = F.softmax(output, dim=0)
        output = element_wise_mul(f_output, output.permute(1,0)).squeeze(1)
        # output.shape = (batch_size, 100)
        output = torch.sigmoid(self.fc(output))

        return output, h_output


class HAN(nn.Module):
    
    def __init__(self, X, embed_size=5, lcl_h_size=50, glb_h_size=50):
                
        super(HAN, self).__init__()

        n_instances, time_len, f_size = X.shape

        embed_num = np.max(X)

        self.batch_size = conf.BATCH_SIZE
        self.lcl_h_size = lcl_h_size
        self.glb_h_size = glb_h_size

        self.lcl_att_net = LocalAttNet(f_size, embed_num, embed_size)
        self.glb_att_net = GlobalAttNet(self.glb_h_size, self.lcl_h_size)
        
        self._init_h_state()

    def _init_h_state(self, last_batch_size=None, init=1):
        if init == 1:  # the first epoch of training
            self.lcl_h_state = nn.Parameter(torch.rand(size=(2, self.batch_size, self.lcl_h_size)), requires_grad=True)
            self.glb_h_state = nn.Parameter(torch.rand(size=(2, self.batch_size, self.glb_h_size)), requires_grad=True)
        
        # if the last_batch_size has been set, then narrow the hidden states
        # of both the local and the global attention
        """if last_batch_size != None and last_batch_size != self.batch_size: 
            self.lcl_h_state.data = torch.narrow(self.lcl_h_state.data, 1, 0, last_batch_size)
            self.glb_h_state.data = torch.narrow(self.glb_h_state.data, 1, 0, last_batch_size)"""


        if conf.GPU == 1:
            self.lcl_h_state = self.lcl_h_state.cuda()
            self.glb_h_state = self.glb_h_state.cuda()

    def forward(self, input):
        output_list = list()
        # input.shape = (batch_size, LAG, f_size)
        input = input.permute(1,0,2)
        # input.shape = (LAG, batch_size, f_size)

        for i in input:
            # i.shape = (batch_size, f_size)
            output, self.lcl_h_state.data = self.lcl_att_net(i, self.lcl_h_state)
            # output.shape = (batch_size, 1, 2 * lcl_h_size)
            # output.permute(1,0,2).shape = (1, batch_size, 2 * lcl_h_size)
            output_list.append(output.permute(1,0,2))

        output = torch.cat(output_list)
        # output.shape = (LAG, batch_size, 2 * lcl_h_size)

        # we need to put the batch_size first. So we repermute the output
        # and shape it as (batch_size, LAG, 2 * lcl_h_size)
        output, self.glb_h_state.data = self.glb_att_net(output.permute(1,0,2),\
                                                    self.glb_h_state)
        return output
