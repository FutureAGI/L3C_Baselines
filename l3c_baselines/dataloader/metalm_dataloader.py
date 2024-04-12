import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class LMDataSet(Dataset):
    def __init__(self, directory, file_size, verbose):
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        self.file_list = os.listdir(directory)
        self.file_list = [os.path.join(directory, file) for file in self.file_list]
        self.file_size = file_size
        self.reset()
        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.index_inverse_list))

    def reset(self):
        random.shuffle(self.file_list)
        self.index_inverse_list = []
        for file in self.file_list:
            self.index_inverse_list.extend([(file, i) for i in range(self.file_size)])

    def __getitem__(self, index):
        path, sub_index = self.index_inverse_list[index]
        data = np.load(path)

        return data[:, :, 0], data[:, :, 1]

    def __len__(self):
        return len(self.index_inverse_list)


# Test Maze Data Set
if __name__=="__main__":
    data_path = sys.argv[1]
    dataset = MazeDataSet(data_path, 1280)
    print("The number of data is: %s" % len(dataset))
    fea, lab = dataset[0]
    print(fea.shape, lab.shape)
