from config import conf

import torch
import torch.nn as nn

import torch.nn.functional as F
from torchnlp.nn import Attention

import numpy as np

class CFIN(nn.Module):

  def __init__(self, X, ctx_size, f_embed_size, ctx_embed_size,
               attn_size, conv_out_c=1, padding=0, stride=1):
    """
      Instantiates a CFIN according to Feng et al. "Understanding Dropouts in
      MOOCs".

      Parameters:
        - f_size (int): size of the feature vector
        - ctx_size (int): size of the context vector
        - f_embed_size (int): size of the embedded feature vector
        - ctx_embed_size (int): size of the embedded context vector
        - attn_size (int): size of the input for the context and query in the
                           attention mechanism
        - conv_out_c (int):  number of output channels for the convolution
        - padding (int): padding for the convolution
        - stride (int): stride for the convolution
    """
    super(CFIN, self).__init__()

    self.f_size = int(np.max(X))+1
    self.ctx_size = ctx_size
    self.ctx_embed_size = ctx_embed_size
    self.f_embed_size = f_embed_size
    self.attn_size = attn_size
    self.batch_size = conf.BATCH_SIZE
    self.conv_out_c = conv_out_c
    self.padding = padding
    self.stride = stride
    self.k = self.f_embed_size*X.shape[-1]*X.shape[-2] + 2*self.padding + stride\
            - self.attn_size*stride


    self._create_ctx()


    self.context_embedding = nn.Embedding(self.ctx_size, self.ctx_embed_size)
    self.fc_context = nn.Linear(self.ctx_embed_size, self.attn_size)


    self.feature_embedding = nn.Embedding(self.f_size, self.f_embed_size)
    # conv1d not conv2d
    self.conv_features = nn.Conv1d(in_channels=1, out_channels=self.conv_out_c,
                                   kernel_size=self.k)

    self.attention = Attention(self.attn_size)

    self.fc = nn.Linear(self.attn_size, 1)
    # flatten the embedded feature matrix
    self.flatten = nn.Flatten(start_dim=2)

  def _create_ctx(self, mean=0, std=0.05):
    self.ctx = nn.Parameter(torch.Tensor(self.batch_size, self.ctx_size, 1), requires_grad=True)
    self.ctx.data.normal_(mean, std)

  # def forward(self, X, ctx):
  def forward(self, X):
    ctx_embed = self.context_embedding(self.ctx.long())#.permute((0,2,1))
    ctx_fc = F.relu(self.fc_context(ctx_embed))
    ctx_fc = ctx_fc.squeeze(2) # get rid of the second dimension = 1

    X_embed = self.feature_embedding(X.long())
    X_flattened = self.flatten(X_embed)
    X_conv = F.relu(self.conv_features(X_flattened))    

    attn_out, attn_weights = self.attention(X_conv, ctx_fc)

    return torch.sigmoid(self.fc(attn_out))

    #return torch.sigmoid(self.fc(X_conv))
