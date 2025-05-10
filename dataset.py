# SET UP DATASET
from io import open
import glob
import os
import time
import torch
from torch.utils.data import Dataset
from preprocess import string_to_tensor

class NamesDataset (Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir # where the data is stored
        self.labels_unique = [] # unique list of every language (use list to preserve order, then dict to remove duplicates)

        self.data = [] # list of names
        self.data_tensors = [] # list of names in tensor lists
        self.labels = [] # list of languages
        self.labels_tensors = [] # list of languages in tensors

        # read all the text files in the directory
        text_files = glob.glob(os.path.join(data_dir, '*.txt')) # make an iterable of all the text files in the data directory
        for filename in text_files: # go thru each of the text files, get the name of the file, add to labels set
            label = os.path.splitext(os.path.basename(filename))[0]
            self.labels_unique.append(label)
            lines = open(filename, encoding='utf-8').read().strip().split('\n') # list of each name in a file
            for name in lines: # go through each name, add the name, name in tensor list, language
                self.data.append(name)
                self.data_tensors.append(string_to_tensor(name))
                self.labels.append(label)

        # go thru the labels and turn them into tensors
        self.labels_unique = list(dict.fromkeys(self.labels_unique)) # remove possible duplicates by turning into dict then back
        for index in range(len(self.labels)):
            temp_tensor = torch.tensor([self.labels_unique.index(self.labels[index])], dtype = torch.long)
            self.labels_tensors.append(temp_tensor)

    def __len__(self):
        return len(self.data)
    
    # return label tensor (language in tensor), data tensor list (name in tensors), label (language), and data (name)
    def __getitem__(self, index:int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        return self.labels_tensors[index], self.data_tensors[index], self.labels[index], self.data[index]
