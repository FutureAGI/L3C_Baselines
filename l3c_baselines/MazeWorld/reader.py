import os
import sys
import lmdb
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class MazeDataSet(Dataset):
    def __init__(self, directory, time_step):
        print("\n\nInitializing data set from file: %s..." % directory)
        self.file_list = os.listdir(directory)
        self.file_list = [os.path.join(directory, file) for file in self.file_list]
        self.time_step = time_step
        self.reset()
        print("...finished initializing data set\n\n")

    def reset(self):
        random.shuffle(self.file_list)
        self.index_inverse_list = []
        for file in self.file_list:
            seq_len = np.load(file + '/actions.npy').shape[0]
            n = seq_len // self.time_step
            self.index_inverse_list.extend([(file, i * self.time_step) for i in range(n)] * seq_len)

    def __getitem__(self, index):
        path, sub_index = self.index_inverse_list[index]

        observations = np.load(path + '/observations.npy')
        actions = np.load(path + '/actions.npy')
        rewards = np.load(path + '/rewards.npy')
        maps = np.load(path + '/maps.npy')
        assert actions.shape[0] == rewards.shape[0] == maps.shape[0] == observations.shape[0] - 1, \
                "The shape of actions = rewards = maps = observations - 1, but get %s, %s, %s and %s" % (actions.shape[0], rewards.shape[0], maps.shape[0], observations.shape[0])

        n_b = sub_index
        n_e = sub_index + self.time_step
        obs_arr = torch.from_numpy(observations[n_b:(n_e + 1)]).float() 
        act_arr = torch.from_numpy(actions[n_b:n_e]).long() 
        rew_arr = torch.from_numpy(rewards[n_b:n_e]).float()
        map_arr = torch.from_numpy(maps[n_b:n_e]).float()

        return obs_arr, act_arr, rew_arr, map_arr

    def __len__(self):
        return len(self.index_inverse_list)


# Test Maze Data Set
if __name__=="__main__":
    data_path = sys.argv[1]
    dataset = MazeDataSet(data_path, 1280)
    print("The number of data is: %s" % len(dataset))
    obs, act, rew, map = dataset[0]
    print(obs.shape, act.shape, rew.shape, map.shape)
