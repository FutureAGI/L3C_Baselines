import argparse
import pickle
import numpy as np
import random
import os
from stable_baselines3 import SAC, PPO
from sb3_contrib import RecurrentPPO
import torch.nn as nn
from l3c.anymdpv2 import AnyMDPv2TaskSampler
from l3c.anymdpv2 import AnyMDPEnv
from noise_distiller import NoiseDistillerWrapper, NoiseDistillerPolicy
from stable_baselines3.common.callbacks import BaseCallback
import gym
from sac_trainer import SACTrainer
from ppo_mlp_trainer import PPO_MLP_Trainer
from ppo_lstm_trainer import PPO_LSTM_Trainer

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        return True

class RLCoach:
    def __init__(self, env, n_epochs, mode, task, seed=None, policies_to_use=None):
        self.env = env
        self.seed = seed
        self.n_epochs = n_epochs
        self.mode = mode
        self.task = task
        
        # 默认使用所有策略
        if policies_to_use is None:
            policies_to_use = ["sac", "ppo_mlp", "ppo_lstm"]
        self.policies_to_use = policies_to_use
        
        # 基本策略 - 随机策略始终可用
        self.policies = {
            "random": lambda x: env.action_space.sample()
        }
        
        # 根据用户选择添加策略
        if "sac" in policies_to_use:
            self.policies["sac"] = SACTrainer(env, seed).model
            
        if "ppo_mlp" in policies_to_use:
            self.policies["ppo_mlp"] = PPO_MLP_Trainer(env, seed).model
            
        if "ppo_lstm" in policies_to_use:
            self.policies["ppo_lstm"] = PPO_LSTM_Trainer(env, seed).model
        
        # 训练器对象 - 用于实际训练
        self.trainers = {}
        if "sac" in policies_to_use:
            self.trainers["sac"] = SACTrainer(env, seed)
            
        if "ppo_mlp" in policies_to_use:
            self.trainers["ppo_mlp"] = PPO_MLP_Trainer(env, seed)
            
        if "ppo_lstm" in policies_to_use:
            self.trainers["ppo_lstm"] = PPO_LSTM_Trainer(env, seed)
        
        self.stages = {
            "random": {"epochs": 2, "policy": "random"},
            "early": {"epochs": 3, "policy": "best_of_all"},   
            "middle": {"epochs": 3, "policy": "best_of_all"},  
            "final": {"epochs": 3, "policy": "best_of_all"},
            "finalnoisedistiller": {"epochs": 3, "policy": "noise_distilled"}
        }
        
        self.policy_snapshots = {stage: [] for stage in self.stages.keys()}

    def preprocess_state(self, state):
        # 如果state是numpy数组,确保其为float32类型
        if isinstance(state, np.ndarray):
            return state.astype(np.float32)
        # 如果是list,转换为numpy数组
        elif isinstance(state, list):
            return np.array(state, dtype=np.float32)
        # 其他情况直接返回
        return state

    def check_env_validity(self, num_steps=10):
        """
        Check if the environment is valid by running 10 steps of RANDOM and policy policies
        and comparing their total rewards.
        
        Returns:
            bool: True if environment is valid, False otherwise
        """
        print("Checking environment validity...")
        
        # Run random policy for num_steps
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        random_rewards = []
        for _ in range(num_steps):
            action = self.policies["random"](state)
            step_result = self.env.step(action)
            if len(step_result) == 5: 
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  
                next_state, reward, done, info = step_result
            random_rewards.append(reward)
            if done:
                break
            state = next_state
        
        # 优先使用LSTM进行对比，如果没有则使用其他可用策略
        compare_policy = None
        if "ppo_lstm" in self.policies_to_use:
            compare_policy = "ppo_lstm"
        elif "ppo_mlp" in self.policies_to_use:
            compare_policy = "ppo_mlp"
        elif "sac" in self.policies_to_use:
            compare_policy = "sac"
        else:
            print("No RL policies available for validation. Considering environment as valid.")
            return True
            
        print(f"Using {compare_policy.upper()} for environment validation")
        
        # Reset and run the chosen policy for num_steps
        state = self.env.reset()
        if isinstance(state, tuple):  # 处理新版本 Gym 的 reset() 返回值
            state = state[0]
        policy_rewards = []
        lstm_states = None
        
        for _ in range(num_steps):
            if compare_policy == "ppo_lstm":
                action, lstm_states = self.policies[compare_policy].predict(
                    state, 
                    state=lstm_states,  
                    deterministic=False
                )
            else:
                action, _ = self.policies[compare_policy].predict(state, deterministic=False)
                
            step_result = self.env.step(action)
            if len(step_result) == 5:  # 新版本 Gym
                next_state, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:  # 旧版本 Gym
                next_state, reward, done, info = step_result
            policy_rewards.append(reward)
            if done:
                break
            state = next_state
        
        # Compare total rewards
        random_total = sum(random_rewards)
        policy_total = sum(policy_rewards)
        
        if policy_total - random_total <= max(3.0 * np.std(random_rewards), 1e-3):
            print(f"Environment invalid: no significant improvements for RL")
            print(f"Random reward: {random_total}, {compare_policy.upper()} reward: {policy_total}")
            return False
        
        print(f"Environment valid - Random={random_total}, {compare_policy.upper()}={policy_total}")
        return True
                
    def train_policies(self, max_steps_per_epoch, max_episodes_per_epoch):
        print("Training policies by stages...")
        
        # 初始化所有算法的 logger
        from stable_baselines3.common.logger import configure
        tmp_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tmp")
        os.makedirs(tmp_path, exist_ok=True)

        for stage, config in self.stages.items():
            print(f"\nTraining stage: {stage}")
            policy_name = config["policy"]

            if policy_name == "random":
                for epoch in range(config["epochs"]):
                    state = self.env.reset()
                    if isinstance(state, tuple):
                        state = state[0]
                    rewards = []
                    done = False

                    while not done:
                        action = self.policies["random"](state)
                        step_result = self.env.step(action)
                        if len(step_result) == 5:
                            next_state, reward, terminated, truncated, info = step_result
                            done = terminated or truncated
                        else:
                            next_state, reward, done, info = step_result
                        rewards.append(reward)
                        state = next_state

                    for epoch in range(config["epochs"]):
                        self.policy_snapshots[stage].append({
                            "stage": stage,
                            "epoch": epoch,
                            "policy_name": "random",
                            "state_dict": None
                        })
                    continue

            elif policy_name == "noise_distilled":
                print("Creating noise distiller policies based on final stage policies...")
                if not self.policy_snapshots["final"]:
                    print("No final stage policies available. Skipping finalnoisedistiller stage.")
                    continue

                for epoch in range(config["epochs"]):
                    base_policy_data = random.choice(self.policy_snapshots["final"])

                    noise_distiller = NoiseDistillerWrapper(
                        self.env, 
                        base_policy_data["state_dict"],
                        max_steps=max_steps_per_epoch
                    )

                    self.policy_snapshots["finalnoisedistiller"].append({
                        "stage": "finalnoisedistiller",
                        "epoch": epoch,
                        "policy_name": f"noise_distilled_{base_policy_data['policy_name']}",
                        "state_dict": base_policy_data["state_dict"],
                        "noise_params": {
                            "upper": noise_distiller.noise_upper,
                            "lower": noise_distiller.noise_lower,
                            "decay_steps": noise_distiller.noise_decay_steps
                        }
                    })
                    print(f"Created noise distiller policy {epoch} based on {base_policy_data['policy_name']}")
                continue

            elif policy_name == "best_of_all":
                for epoch in range(config["epochs"]):
                    print(f"\nEpoch {epoch} in stage {stage}:")
                    results = {}

                    # 只训练用户选择的策略
                    for p_name in self.policies_to_use:
                        try:
                            print(f"Training {p_name.upper()}...")
                            # 使用训练器进行训练
                            train_result = self.trainers[p_name].train(
                                episodes=max_episodes_per_epoch, 
                                max_steps=max_steps_per_epoch
                            )
                            
                            results[p_name] = train_result
                            
                            avg_return = train_result["avg_return"]
                            episode_count = train_result["episode_count"]
                            total_steps = train_result["total_steps"]
                            total_success = train_result["total_success"]
                            
                            print(f"{p_name.upper()} - Epoch {epoch}: episodes={episode_count}, steps={total_steps}, "
                                f"success_rate={(total_success/episode_count if episode_count > 0 else 0):.2%}, "
                                f"avg_return={avg_return:.2f}")
                        
                        except Exception as e:
                            print(f"Unexpected error for {p_name}: {e}")
                            import traceback
                            traceback.print_exc()
                            continue

                    if results:  
                        # 选择表现最好的策略
                        best_policy = max(results.keys(), key=lambda k: results[k]["avg_return"])
                        # 获取策略的状态字典
                        state_dict = self.trainers[best_policy].get_state_dict()

                        self.policy_snapshots[stage].append({
                            "stage": stage,
                            "epoch": epoch,
                            "policy_name": best_policy,
                            "state_dict": state_dict
                        })
                        print(f"Epoch {epoch} best policy: {best_policy.upper()} with avg_return: {results[best_policy]['avg_return']:.2f}")
                    else:
                        print("No policies successfully completed training in this epoch")
            
            else:
                print(f"Unknown policy type: {policy_name}")

        return True

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        behavior_policies = {}
        reference_policies = {}
        
        for stage in self.stages.keys():
            if len(self.policy_snapshots[stage]) > 0:
                behavior_policies[stage] = self.policy_snapshots[stage]  

        if len(self.policy_snapshots["final"]) > 0:
            reference_policies["final"] = self.policy_snapshots["final"][-1]
        
        save_data = {
            "mode": self.mode,  
            "task_config": {    
                "mode": self.mode,
                "ndim": self.task.ndim if hasattr(self.task, 'ndim') else None,
                "state_dim": self.task.state_dim if hasattr(self.task, 'state_dim') else None,
                "action_dim": self.task.action_dim if hasattr(self.task, 'action_dim') else None
            },
            "behavior_policies": behavior_policies,
            "reference_policies": reference_policies
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="directory to save the trained coaches")
    parser.add_argument("--mode", type=str, required=False, choices=["static", "dynamic", "universal"], help="Mode for task sampler.")
    parser.add_argument("--state_dim", type=int, default=256, help="state dimension")
    parser.add_argument("--action_dim", type=int, default=256, help="action dimension")
    parser.add_argument("--ndim", type=int, default=8, help="ndim for task sampler")
    parser.add_argument("--max_steps", type=int, default=4000, help="maximum steps per epoch")
    parser.add_argument("--max_episodes", type=int, default=20, help="maximum episodes per epoch")
    parser.add_argument("--n_epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--policy", type=str, nargs='+', choices=["sac", "ppo_mlp", "ppo_lstm"], 
                        help="Specify which policies to use. If not specified, all policies will be used.")
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    if args.mode is not None:
        modes = [args.mode]
    else:
        modes = ["static", "dynamic", "universal"]
    
    for mode in modes:
        print(f"\nTraining coach for mode: {mode} ")
        
        training_successful = False
        max_attempts = 100  
        attempt = 0
        
        while not training_successful and attempt < max_attempts:
            attempt += 1
            print(f"Attempt {attempt} for mode {mode}")
            
            env = gym.make("anymdp-v2-visualizer")
            task = AnyMDPv2TaskSampler(
                state_dim=args.state_dim,
                action_dim=args.action_dim,
                seed=args.seed
            )
            env.set_task(task)
            
            coach = RLCoach(env, args.n_epochs, mode=mode, seed=args.seed, task=task, policies_to_use=args.policy)
            
            # Check environment validity before training
            env_valid = coach.check_env_validity(num_steps=10)
            if not env_valid:
                print("This environment is invalid because RANDOM and RL policies produce similar rewards. Rebuilding env and task...")
                continue
                
            # If we reach here, the environment is valid, so proceed with training
            training_successful = coach.train_policies(args.max_steps, args.max_episodes)
            if not training_successful:
                print("Training unsuccessful. Rebuilding env and task...")
            else:
                print("Task available. Training completed successfully.")
        
        if not training_successful:
            print(f"After attempting {max_attempts} times, mode {mode} fails to generate available task. Skipping this mode.")
            continue
        
        save_path = os.path.join(args.save_dir, f"coach_{mode}.pkl")
        coach.save(save_path)