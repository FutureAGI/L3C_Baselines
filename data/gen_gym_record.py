import argparse
import os
import multiprocessing
import numpy as np

import gymnasium as gym
from gym.envs.toy_text.frozen_lake import generate_random_map

from stable_baselines3 import DQN, A2C, TD3, PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def create_env(env_name):
    if(env_name.lower() == "lake"):
        env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), is_slippery=True)
        return env
    elif(env_name.lower() == "lander"):
        env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)
        return env
    else:
        raise ValueError("Unknown env name: {}".format(env_name))
        
def train_model(env, model_name, n_total_timesteps, save_path):
    # For more RL alg, please refer to https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
    if(model_name.lower() == "dqn"):
        model = DQN('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=int(n_total_timesteps))
        file_path = f'{save_path}/model/dqn'
        create_directory(file_path)
        model.save(file_path)
        return model
    elif(model_name.lower() == "a2c"):
        model = A2C('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=int(n_total_timesteps))
        file_path = f'{save_path}/model/a2c'
        create_directory(file_path)
        model.save(file_path)
        return model
    elif(model_name.lower() == "td3"):
        model = TD3('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=int(n_total_timesteps))
        file_path = f'{save_path}/model/td3'
        create_directory(file_path)
        model.save(file_path)
        return model
    elif(model_name.lower() == "ppo"):
        model = PPO('MlpPolicy', env, verbose=1)
        model.learn(total_timesteps=int(n_total_timesteps))
        file_path = f'{save_path}/model/ppo'
        create_directory(file_path)
        model.save(file_path)
        return model
    else:
        raise ValueError("Unknown policy type: {}".format(model_name))

def load_model(model_name, save_path, env):
    if(model_name.lower() == "dqn"):
        model = DQN.load(f'{save_path}/model/dqn.zip', env=env)
        return model
    elif(model_name.lower() == "a2c"):
        model = A2C.load(f'{save_path}/model/a2c.zip', env=env)
        return model
    elif(model_name.lower() == "td3"):
        model = TD3.load(f'{save_path}/model/td3.zip', env=env)
        return model
    elif(model_name.lower() == "ppo"):
        model = PPO.load(f'{save_path}/model/ppo.zip', env=env)
        return model
    else:
        raise ValueError("Unknown policy type: {}".format(model_name))

def produce_data(env_name, model_name, save_path, epoch_ids, max_try=100):
    # Create environment
    env = create_env(env_name)
    env = DummyVecEnv([lambda: env])  # Wrap the environment as a vectorized environment

    model = load_model(model_name, save_path, env)

    task_count = 0
    success_count = 0
    total_action_count = 0

    for idx in epoch_ids:
        state = env.reset()
        done = False
        action_count = 0
        total_reward = 0
        
        # Initialize lists to store states, actions, and rewards
        state_list = []
        act_list = []
        reward_list = []

        while not done and action_count < max_try:
            action, _ = model.predict(state)  # Select action
            next_state, reward, done, _ = env.step(action)  # Execute action
            
            # Record state, action, and reward
            state_list.append(state)  # Append current state
            act_list.append(action)    # Append action taken
            reward_list.append(reward) # Append reward received
            
            total_reward += reward
            state = next_state
            action_count += 1

        # Print action count and total reward
        print(f"Epoch {idx}: Action count = {action_count}, Total reward = {total_reward}")
        task_count += 1 
        if(env_name.lower() == "lake"):
            if total_reward > 0:
                success_count += 1
                total_action_count += action_count
        elif(env_name.lower() == "lander"):
            if total_reward > 200:
                success_count += 1
                total_action_count += action_count

        # Create save directory
        file_path = f'{save_path}/data/record-{idx:06d}'
        create_directory(file_path)

        # Save data
        results = {
            "states": np.array(state_list, dtype=np.uint32),
            "actions": np.array(act_list, dtype=np.uint32),
            "rewards": np.array(reward_list, dtype=np.float32)
        }
        
        np.save(f"{file_path}/observations.npy", results["states"])  # Save state data
        np.save(f"{file_path}/actions_behavior.npy", results["actions"])  # Save action data
        np.save(f"{file_path}/rewards.npy", results["rewards"])  # Save reward data
    
    if(env_name.lower() == "lake" or env_name.lower() == "lander"):
      print(f"Epoch {idx}: average action count when success = {total_action_count/success_count}, success rate = {success_count/task_count}")
    
    env.close()

def worker(worker_id, env_name, model_name, save_path, epoch_ids, max_try):
    # Call produce_data function
    produce_data(env_name, model_name, save_path, epoch_ids, max_try)

if __name__ == "__main__":
    # Use argparse to parse command line arguments
    parser = argparse.ArgumentParser(description="Train a Q-learning agent in a gym environment.")
    parser.add_argument('--env_name', choices=['LAKE', 'LANDER'], default='LAKE', help="The name of the gym environment")
    parser.add_argument('--save_path', type=str, required=True, help='The path to save the training data (without file extension).')
    parser.add_argument('--policy_name', choices=['DQN', 'A2C', 'TD3', 'PPO'], default='DQN', help="Policy Type")
    parser.add_argument('--n_total_timesteps', type=int, default=200000, help='Total number of epochs for training.')
    parser.add_argument('--n_task', type=int, default=1000, help='Total number of task for generating.')
    parser.add_argument('--n_max_try', type=int, default=100, help='Maximum number of actions per epoch.')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers for training.')
    parser.add_argument('--enable_load_model', type=str, default="False", help='Whether to load a pre-trained model.')

    args = parser.parse_args()
    args.enable_load_model = args.enable_load_model.lower() == "true"

    env = create_env(args.env_name)
    if not args.enable_load_model:
      model = train_model(env, args.policy_name, args.n_total_timesteps, args.save_path)

    # Calculate the number of tasks for each worker process
    worker_splits = args.n_task / args.n_workers + 1.0e-6
    processes = []
    n_b_t = 0

    for worker_id in range(args.n_workers):
        n_e_t = n_b_t + worker_splits
        n_b = int(n_b_t)
        n_e = int(n_e_t)
        print("Start processes generating %04d to %04d" % (n_b, n_e))
        
        # Create epoch_ids
        epoch_ids = range(n_b, n_e)
        multiprocessing.set_start_method('spawn', force=True)
        process = multiprocessing.Process(target=worker, 
                                          args=(worker_id, args.env_name, args.policy_name, args.save_path, epoch_ids, args.n_max_try))
        processes.append(process)
        process.start()
        n_b_t = n_e_t

    for process in processes:
        process.join()

    print("Training completed, data has been saved.")