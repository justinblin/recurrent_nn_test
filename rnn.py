import torch

# use GPU if possible
device = torch.device('cpu')
if torch.cuda.is_available():
    decide = torch.device('cuda')
torch.set_default_device(device)

print(device)