import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from l3c_baselines.utils import rewards2go


class AnyMDPDataSet(Dataset):
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

    def reset(self, seed=0):
        random.seed(seed)
        random.shuffle(self.file_list)


    def __getitem__(self, index):
        path = self.file_list[index]

        try:
            observations = np.load(path + '/observations.npy')
            actions_behavior = np.load(path + '/actions_behavior.npy')
            actions_label = np.load(path + '/actions_label.npy')
            rewards = np.load(path + '/rewards.npy')
            max_t = min(actions_label.shape[0], 
                        rewards.shape[0], 
                        actions_behavior.shape[0],
                        observations.shape[0])

            # Shape Check
            if(self.time_step > max_t):
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step
            obs_arr = torch.from_numpy(observations[n_b:n_e].astype("int32")).long() 
            bact_arr = torch.from_numpy(actions_behavior[n_b:n_e].astype("int32")).long() 
            lact_arr = torch.from_numpy(actions_label[n_b:n_e].astype("int32")).long() 
            reward_arr = torch.from_numpy(rewards[n_b:n_e]).float()

            return obs_arr, bact_arr, lact_arr, reward_arr
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None

    def __len__(self):
        return len(self.file_list)


# Test Maze Data Set
if __name__=="__main__":
    data_path = sys.argv[1]
    dataset = AnyMDPDataSet(data_path, 1280, verbose=True)
    print("The number of data is: %s" % len(dataset))
    obs, bact, lact, rewards = dataset[0]
    print(obs.shape, bact.shape, lact.shape, rewards.shape)
