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

def create_env(env_name, randon_env):
    if(env_name.lower() == "lake"):
        if randon_env:
            env = gym.make('FrozenLake-v1', desc=generate_random_map(size=4), is_slippery=True)
            return env
        else:
            env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
            return env
    elif(env_name.lower() == "lander"):
        env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)
        return env
    else:
        raise ValueError("Unknown env name: {}".format(env_name))

def train_model(env, model_name, n_total_timesteps, save_path):
    # For more RL alg, please refer to https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
    model_classes = {'dqn': DQN, 'a2c': A2C, 'td3': TD3, 'ppo': PPO}
    if model_name.lower() not in model_classes:
        raise ValueError("Unknown policy type: {}".format(model_name))
    
    model = model_classes[model_name.lower()]('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=int(n_total_timesteps))
    file_path = f'{save_path}/model/{model_name.lower()}'
    create_directory(file_path)
    model.save(file_path)
    
    return model

def load_model(model_name, save_path, env):
    model_classes = {'dqn': DQN, 'a24': A2C, 'td3': TD3, 'ppo': PPO}
    if model_name.lower() not in model_classes:
        raise ValueError("Unknown policy type: {}".format(model_name))
    
    return model_classes[model_name.lower()].load(f'{save_path}/model/{model_name.lower()}.zip', env=env)

def produce_data(args, worker_id, shared_list, seg_len):
    # Create environment
    env = create_env(args.env_name, args.random_env)
    env = DummyVecEnv([lambda: env])  # Wrap the environment as a vectorized environment

    model = load_model(args.model_name, args.save_path, env)

    state_list = []
    act_list = []
    reward_list = []

    task_count = 0
    success_count = 0
    total_action_count = 0
    step = 0
    while step < seg_len:
        trail_reward = 0
        state = env.reset()
        done = False
        step_trail_start = step
        while not done:
            action, _ = model.predict(state)  # Select action
            next_state, reward, done, _ = env.step(action)  # Execute action
            
            # Record state, action, and reward
            state_list.append(state)  # Append current state
            act_list.append(action)    # Append action taken
            reward_list.append(reward) # Append reward received
            
            if done:
                state_list.append(next_state)  # Append next state
                act_list.append(args.action_done)    # Append action flag
                reward_list.append(args.reward_done) # Append reward zero

            trail_reward += reward
            state = next_state
            step += 1
        task_count += 1
        if(args.env_name.lower() == "lake"):
            if trail_reward > 0:
                success_count += 1
                total_action_count += step - step_trail_start
        elif(args.env_name.lower() == "lander"):
            if trail_reward > 200:
                success_count += 1
                total_action_count += step - step_trail_start

    if(args.env_name.lower() == "lake" or args.env_name.lower() == "lander"):
      print(f"Worker {worker_id}: average action count when success = {total_action_count/success_count}, success rate = {success_count/task_count}")

    result = {
        "states": np.squeeze(np.array(state_list)),
        "actions": np.squeeze(np.array(act_list)),
        "rewards": np.squeeze(np.array(reward_list))
    }
    env.close()
    shared_list.append(result)
    return

def generate_records(args, task_id):
    env = create_env(args.env_name, args.random_env)
    if not args.enable_load_model or args.random_env:
        train_model(env, args.policy_name, args.n_total_timesteps, args.save_path)

    worker_splits = args.n_seq_len / args.n_workers + 1.0e-6
    manager = multiprocessing.Manager()
    shared_list = manager.list()
    processes = []

    for worker_id in range(args.n_workers):

        multiprocessing.set_start_method('spawn', force=True)
        process = multiprocessing.Process(target=produce_data, 
                                          args=(args, worker_id, shared_list, worker_splits))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
    
    results = list(shared_list)
    merged_result = {}
    for result in results:
        if "states" not in merged_result:
            merged_result["states"] = result["states"]
            merged_result["actions"] = result["actions"]
            merged_result["rewards"] = result["rewards"]
            continue            
        merged_result["states"] = np.concatenate((merged_result["states"], result["states"]))
        merged_result["actions"] = np.concatenate((merged_result["actions"], result["actions"]))
        merged_result["rewards"] = np.concatenate((merged_result["rewards"], result["rewards"]))
    merged_result["states"] = merged_result["states"][:args.n_seq_len]
    merged_result["actions"] = merged_result["actions"][:args.n_seq_len]
    if args.env_name.lower() == "lander":
        merged_result["rewards"] = np.multiply(merged_result["rewards"][:args.n_seq_len], 0.01)
    else:
        merged_result["rewards"] = merged_result["rewards"][:args.n_seq_len]

    file_path = f'{args.save_path}/data/record-{task_id:06d}'
    create_directory(file_path)
    np.save(f"{file_path}/observations.npy", merged_result["states"])  # Save state data
    np.save(f"{file_path}/actions_behavior.npy", merged_result["actions"])  # Save action data
    np.save(f"{file_path}/rewards.npy", merged_result["rewards"])  # Save reward data
    np.save(f"{file_path}/actions_label.npy", merged_result["actions"])  # Save fake label data.




if __name__ == "__main__":
    # Use argparse to parse command line arguments
    parser = argparse.ArgumentParser(description="Train a Q-learning agent in a gym environment.")
    parser.add_argument('--env_name', choices=['LAKE', 'LANDER'], default='LAKE', help="The name of the gym environment")
    parser.add_argument('--save_path', type=str, required=True, help='The path to save the training data (without file extension).')
    parser.add_argument('--policy_name', choices=['DQN', 'A2C', 'TD3', 'PPO'], default='DQN', help="Policy Type")
    parser.add_argument('--n_total_timesteps', type=int, default=200000, help='Total number of epochs for training.')
    parser.add_argument('--n_task', type=int, default=1000, help='Total number of task for generating.')
    parser.add_argument('--n_seq_len', type=int, default=100, help='Maximum number of actions per epoch.')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of parallel workers for training.')
    parser.add_argument('--enable_load_model', type=str, default="False", help='Whether to load a pre-trained model.')
    parser.add_argument('--random_env', type=str, default="False", help='Random env.')
    parser.add_argument('--action_done', type=int, default=4, help='Action when done.')
    parser.add_argument('--reward_done', type=float, default=0.0, help='Reward when left.')


    args = parser.parse_args()
    args.enable_load_model = args.enable_load_model.lower() == "true"
    args.random_env = args.random_env.lower() == "true"

    for task_id in range(args.n_task):
        generate_records(args, task_id)

    print("Finish generation.")