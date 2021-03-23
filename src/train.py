from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
import torchvision as tv
import torchvision.transforms as tvtf
import scipy.io
from collections import OrderedDict
from torch.autograd import Variable, Function
from train_utils import get_dataloader
from model import HyperspecAE
from train_objectives import SAD, SID

# ------------------ Define HyperParameters ----------------- #
BATCH_SIZE = 20 # Recommended for Samson
LR = 1e-3
num_bands = 156
end_members = 3
threshold = 5 # For Soft Thresholding
DIR = '../Samson/'
dropout = 0.5
EPOCHS = 200
activation = 'LReLU' # Options: [ReLU, LReLU, Sigmoid] -> DeepAE, 'None' -> ShallowAE  
ae_type = 'deep' # Options: [deep, shallow]
OBJECTIVE = 'SAD' # Options: [MSE, SAD, SID]

# ------------------ Training -------------------- #
# Load Data
train_dataloader, test_set = get_dataloader(BATCH_SIZE=BATCH_SIZE, DIR=DIR)

max_batches = len(train_dataloader)

# Define Model
model = HyperspecAE(num_bands, end_members, dropout, activation,
              threshold, ae_type)
optimizer = optim.Adam(model.parameters(), LR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(EPOCHS):
    
    iterator = iter(train_dataloader)

    for batch__ in range(max_batches):
        
        X, _ = next(iterator)
        X = X.view(X.size()[0], -1)
        X = X.cuda()

        enc_out, dec_out = model(X.float())
        
        if OBJECTIVE=="SAD":
            reconstr_loss = SAD()
        elif OBJECTIVE=='MSE':
            reconstr_loss = Snn.MSELoss()
        else:
            reconstr_loss = SID()
        loss = reconstr_loss(dec_out, X.float())
        
        loss = torch.sum(loss).float()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch+1)%50==0:
      print(f'Epoch {epoch + 1:04d} / {EPOCHS:04d}', end='\n=================\n')
      print("Loss: %.4f" %(loss.item()))   
      
print('Training Finished!')