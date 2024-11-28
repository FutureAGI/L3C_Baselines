import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class AnyMDPDataSetBase(Dataset):
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

    def __len__(self):
        return len(self.file_list)

    def _load_and_process_data(self, path):
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

            return observations[n_b:n_e], actions_behavior[n_b:n_e], actions_label[n_b:n_e], rewards[n_b:n_e]
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None, None, None, None

class AnyMDPDataSet(AnyMDPDataSetBase):
    def __getitem__(self, index):
        path = self.file_list[index]

        observations, actions_behavior, actions_label, rewards = self._load_and_process_data(path)
        
        if any(arr is None for arr in [observations, actions_behavior, actions_label, rewards]):
            return None

        obs_arr = torch.from_numpy(observations.astype("int32")).long() 
        bact_arr = torch.from_numpy(actions_behavior.astype("int32")).long() 
        lact_arr = torch.from_numpy(actions_label.astype("int32")).long() 
        reward_arr = torch.from_numpy(rewards).float()

        return obs_arr, bact_arr, lact_arr, reward_arr

class AnyMDPDataSetContinuousState(AnyMDPDataSetBase):
    def __getitem__(self, index):
        path = self.file_list[index]

        observations, actions_behavior, actions_label, rewards = self._load_and_process_data(path)
        
        if any(arr is None for arr in [observations, actions_behavior, actions_label, rewards]):
            return None

        obs_arr = torch.from_numpy(observations).float() 
        bact_arr = torch.from_numpy(actions_behavior.astype("int32")).long() 
        lact_arr = torch.from_numpy(actions_label.astype("int32")).long() 
        reward_arr = torch.from_numpy(rewards).float()

        return obs_arr, bact_arr, lact_arr, reward_arr
    
class AnyMDPDataSetContinuousStateAction(AnyMDPDataSetBase):
    def __getitem__(self, index):
        path = self.file_list[index]

        observations, actions_behavior, actions_label, rewards = self._load_and_process_data(path)
        
        if any(arr is None for arr in [observations, actions_behavior, actions_label, rewards]):
            return None

        obs_arr = torch.from_numpy(observations).float() 
        bact_arr = torch.from_numpy(actions_behavior).float() 
        lact_arr = torch.from_numpy(actions_label).float() 
        reward_arr = torch.from_numpy(rewards).float()

        return obs_arr, bact_arr, lact_arr, reward_arr

# Test Maze Data Set
if __name__=="__main__":
    data_path = sys.argv[1]
    dataset = AnyMDPDataSet(data_path, 1280, verbose=True)
    print("The number of data is: %s" % len(dataset))
    obs, bact, lact, rewards = dataset[0]
    if obs is not None:
        print(obs.shape, bact.shape, lact.shape, rewards.shape)
    else:
        print("Failed to load the first sample.")