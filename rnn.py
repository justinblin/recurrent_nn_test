import torch
import string
import unicodedata

# use GPU if possible
device = torch.device('cpu')
if torch.cuda.is_available():
    decide = torch.device('cuda')
torch.set_default_device(device)


# process names from unicode into ascii
allowed_char = string.ascii_letters + ' .,;\'_'
def unicode_to_ascii(unicode_string):
    return ''.join(curr_char for curr_char in unicodedata.normalize('NFD', unicode_string) 
                   if unicodedata.category != 'Mn' and curr_char in allowed_char)

print(unicode_to_ascii('Ślusàrski'))