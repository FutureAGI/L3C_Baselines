import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class MazeDataSet(Dataset):
    def __init__(self, directory, time_step, verbose):
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        self.file_list = os.listdir(directory)
        self.file_list = [os.path.join(directory, file) for file in self.file_list]
        self.time_step = time_step
        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def reset(self):
        random.shuffle(self.file_list)

    def __getitem__(self, index):
        path = self.file_list[index]

        observations = np.load(path + '/observations.npy')
        actions = np.load(path + '/actions.npy')
        rewards = np.load(path + '/rewards.npy')
        maps = np.load(path + '/maps.npy')
        max_t = actions.shape[0]
        assert max_t == rewards.shape[0] and max_t == maps.shape[0] and max_t + 1 == observations.shape[0], \
                "The 0 dimension shape of actions == rewards == maps == observations - 1, but get %s, %s, %s and %s" % \
                (max_t, rewards.shape[0], maps.shape[0], observations.shape[0])

        if(self.time_step > max_t):
            print('[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
            n_b = 0
            n_e = max_t
        else:
            n_b = random.randint(0, max_t - self.time_step)
            n_e = n_b + self.time_step
        obs_arr = torch.from_numpy(observations[n_b:(n_e + 1)]).float() 
        act_arr = torch.from_numpy(actions[n_b:n_e]).long() 
        rew_arr = torch.from_numpy(rewards[n_b:n_e]).float()
        map_arr = torch.from_numpy(maps[n_b:n_e]).float()

        return obs_arr, act_arr, rew_arr, map_arr

    def __len__(self):
        return len(self.file_list)


# Test Maze Data Set
if __name__=="__main__":
    data_path = sys.argv[1]
    dataset = MazeDataSet(data_path, 1280)
    print("The number of data is: %s" % len(dataset))
    obs, act, rew, map = dataset[0]
    print(obs.shape, act.shape, rew.shape, map.shape)
