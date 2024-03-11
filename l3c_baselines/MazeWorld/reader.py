import sys
import lmdb
import numpy as np

def read_data(path):
    observations = np.load(path + '/observations.npy')
    actions = np.load(path + '/actions.npy')
    rewards = np.load(path + '/rewards.npy')
    maps = np.load(path + '/maps.npy')

    return observations, actions, rewards, maps

if __name__=="__main__":
    data_path = sys.argv[1]
    observations, actions, rewards, maps = read_data(data_path)
    print("observations shape:", observations.shape)
    print("actions shape:", actions.shape)
    print("rewards shape:", rewards.shape)
    print("maps shape:", maps.shape)
