# METHODS TO PREPROCESS DATA INTO TENSORS
import torch
import string
import unicodedata

# process names from unicode into ascii
allowed_char = string.ascii_letters + ' .,;\'_'
def unicode_to_ascii(unicode_string):
    return ''.join(curr_char for curr_char in unicodedata.normalize('NFD', unicode_string) 
                   if unicodedata.category != 'Mn' and curr_char in allowed_char)

# turn letter into index for one hot tensor
def letter_to_index(letter): 
    index = allowed_char.find(letter)
    return len(allowed_char)-1 if index == -1 else index

# turn name into list of one hot tensors
def string_to_tensor(line): 
    tensor = torch.zeros(len(line), 1, len(allowed_char))
    for index, letter in enumerate(line):
        tensor[index][0][letter_to_index(letter)] = 1
    return tensor