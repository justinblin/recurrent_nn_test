# SET UP BASICS
import torch
import os
from dataset import NamesDataset

# use GPU if possible
device = torch.device('cpu')
if torch.cuda.is_available():
    decide = torch.device('cuda')
torch.set_default_device(device)

# set up dataset
alldata = NamesDataset("data/names")
print(f'loaded {len(alldata)} names')
print(f'example: {alldata[0]}')
print(alldata.labels_unique)