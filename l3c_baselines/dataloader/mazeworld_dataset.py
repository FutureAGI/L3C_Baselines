import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class MazeDataSet(Dataset):
    def __init__(self, directory, time_step, verbose=False):
        if(verbose):
            print("\nInitializing data set from file: %s..." % directory)
        self.file_list = []
        directories = []
        if(isinstance(directory, list)):
            directories.extend(directory)
        else:
            directories.append(directory)
        for d in directories:
            file_list = os.listdir(d)
            self.file_list.extend([os.path.join(d, file) for file in file_list])
            
        self.time_step = time_step
        self.reset()

        if(verbose):
            print("...finished initializing data set, number of samples: %s\n" % len(self.file_list))

    def reset(self):
        random.shuffle(self.file_list)

    def __getitem__(self, index):
        path = self.file_list[index]

        observations = np.load(path + '/observations.npy')
        actions_behavior = np.load(path + '/actions_behavior.npy')
        actions_label = np.load(path + '/actions_label.npy')
        rewards = np.load(path + '/rewards.npy')
        targets = np.load(path + '/targets_location.npy')
        max_t = actions_behavior.shape[0]
        assert max_t == rewards.shape[0] and max_t == actions_label.shape[0] and max_t + 1 == observations.shape[0], \
                "The 0 dimension shape of actions == rewards == label == observations - 1, but get %s, %s, %s and %s" % \
                (max_t, rewards.shape[0], rewards.shape[0], actions_label.shape[0], observations.shape[0])

        if(self.time_step > max_t):
            print('[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
            n_b = 0
            n_e = max_t
        else:
            n_b = 0 #random.randint(0, max_t - self.time_step)
            n_e = n_b + self.time_step
        obs_arr = torch.from_numpy(observations[n_b:(n_e + 1)]).float() 
        bact_arr = torch.from_numpy(actions_behavior[n_b:n_e]).long() 
        lact_arr = torch.from_numpy(actions_label[n_b:n_e]).long() 
        rew_arr = torch.from_numpy(rewards[n_b:n_e]).float()
        target_arr = torch.from_numpy(targets[n_b:n_e]).float()

        return obs_arr, bact_arr, lact_arr, rew_arr, target_arr

    def __len__(self):
        return len(self.file_list)


# Test Maze Data Set
if __name__=="__main__":
    data_path = sys.argv[1]
    dataset = MazeDataSet(data_path, 1280, verbose=True)
    print("The number of data is: %s" % len(dataset))
    obs, bact, lact, rew, targets = dataset[0]
    print(obs.shape, bact.shape, lact.shape, rew.shape, targets.shape)
