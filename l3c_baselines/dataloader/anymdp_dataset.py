import os
import sys
import random
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset


class AnyMDPDataSet(Dataset):
    def __init__(self, directory, time_step, data_type, verbose=False):
        if verbose:
            print("\nInitializing data set from file: %s..." % directory)
        
        self.file_list = []
        directories = []
        if isinstance(directory, list):
            directories.extend(directory)
        else:
            directories.append(directory)
        
        for d in directories:
            file_list = os.listdir(d)
            self.file_list.extend([os.path.join(d, file) for file in file_list])
            
        self.time_step = time_step
        
        # Determine data types and conversion functions based on data_type
        if data_type.state == "Discrete":
            self.load_observation = lambda obs: torch.from_numpy(obs.astype("int32")).long()
        else:
            self.load_observation = lambda obs: torch.from_numpy(obs).float()
        
        if data_type.action == "Discrete":
            self.load_behavior_action = lambda act: torch.from_numpy(act.astype("int32")).long()
            self.load_label_action = lambda act: torch.from_numpy(act.astype("int32")).long()
        else:
            self.load_behavior_action = lambda act: torch.from_numpy(act).float()
            self.load_label_action = lambda act: torch.from_numpy(act).float()
        
        if data_type.reward == "Discrete":
            self.load_reward = lambda reward: torch.from_numpy(reward.astype("int32")).long()
        else:
            self.load_reward = lambda reward: torch.from_numpy(reward).float()
        
        self.reset()

        if verbose:
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
            if self.time_step > max_t:
                print(f'[Warning] Load samples from {path} that is shorter ({max_t}) than specified time step ({self.time_step})')
                n_b = 0
                n_e = max_t
            else:
                n_b = 0
                n_e = self.time_step

            obs_arr = self.load_observation(observations[n_b:n_e])
            bact_arr = self.load_behavior_action(actions_behavior[n_b:n_e])
            lact_arr = self.load_label_action(actions_label[n_b:n_e])
            reward_arr = self.load_reward(rewards[n_b:n_e])

            return obs_arr, bact_arr, lact_arr, reward_arr
        except Exception as e:
            print(f"Unexpected reading error founded when loading {path}: {e}")
            return None

    def __len__(self):
        return len(self.file_list)


# Test Maze Data Set
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python anymdp_dataset.py <data_path>")
        sys.exit(1)
    
    data_path = sys.argv[1]
    # Convert the dictionary to a simple object with attributes for compatibility with __init__
    class DataTypeConfig:
        def __init__(self, config):
            self.state = config['state']
            self.action = config['action']
            self.reward = config['reward']

    data_type_config_dict = {
        "prompt": "Continuous",  # Example configuration
        "state": "Discrete",
        "action": "Discrete",
        "reward": "Continuous"
    }
    data_type = DataTypeConfig(data_type_config_dict)
    dataset = AnyMDPDataSet(data_path, 1280, data_type, verbose=True)
    print("The number of data is: %s" % len(dataset))
    obs, bact, lact, rewards = dataset[0]
    print(obs.shape, bact.shape, lact.shape, rewards.shape)