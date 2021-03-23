from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from collections import OrderedDict
from torch.autograd import Variable, Function
from layers import GaussianDropout, ASC

class HyperspecAE(nn.Module):
    def __init__(self, num_bands: int=156, end_members: int=3, dropout: float=1.0,
                 activation: str='ReLU', threshold: int=5, ae_type: str='deep'):
      # Constructor
      super(HyperspecAE, self).__init__()

      if activation == 'ReLU':
        self.act = nn.ReLU()
      elif activation == 'LReLU':
        self.act = nn.LeakyReLU()
      else:
        self.act = nn.Sigmoid()

      self.gauss = GaussianDropout(dropout)
      self.asc = ASC()
      if ae_type == 'deep':
            self.encoder = nn.Sequential(OrderedDict([
                                                ('hidden_1', nn.Linear(num_bands, 9*end_members)),
                                                ('activation_1', self.act),
                                                ('hidden_2', nn.Linear(9*end_members, 6*end_members)),
                                                ('activation_2', self.act),
                                                ('hidden_3', nn.Linear(6*end_members, 3*end_members)),
                                                ('activation_3', self.act),
                                                ('hidden_4', nn.Linear(3*end_members, end_members)),
                                                ('activation_4', self.act),
                                                ('batch_norm', nn.BatchNorm1d(end_members)),
                                                ('soft_thresholding', nn.Softplus(threshold=threshold)),
                                                ('ASC', self.asc),
                                                ('Gaussian_Dropout', self.gauss)

            ]))
      elif ae_type == 'shallow':
            self.encoder = nn.Sequential(OrderedDict([
                                                ('hidden_1', nn.Linear(num_bands, end_members)),
                                                ('batch_norm', nn.BatchNorm1d(end_members)),
                                                ('soft_thresholding', nn.Softplus(threshold=threshold)),
                                                ('ASC', self.asc),
                                                ('Gaussian_Dropout', self.gauss)
            ]))

      self.decoder = nn.Linear(end_members, num_bands)

    def forward(self, img):
      encoded = self.encoder(img)
      decoded = self.decoder(encoded)
      return encoded, decoded