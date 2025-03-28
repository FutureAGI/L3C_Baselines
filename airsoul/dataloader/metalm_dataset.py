import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class LMDataSet(Dataset):
    def __init__(self, directory, max_length, verbose=False):
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        self.file_list = []
        self.max_length = max_length
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        for d in directories:
            file_list = os.listdir(d)
            self.file_list.extend([os.path.join(d, file) for file in file_list])
        self.data_list = []
        for file in self.file_list:
            data = np.load(file)
            assert data.ndim == 3 and data.shape[1] == 2, \
                    f"Expect the data shape of meta_lm being (Bsz, 2, Length), get {data.shape}"
            file_size = data.shape[0]
            self.data_list.extend([(file, i) for i in range(file_size)])
        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.data_list))

    def __getitem__(self, index):
        path, sub_index = self.data_list[index]
        data = np.load(path)
        return torch.from_numpy(data[sub_index][0]).to(torch.int64), torch.from_numpy(data[sub_index][1]).to(torch.int64)

    def __len__(self):
        return len(self.data_list)
