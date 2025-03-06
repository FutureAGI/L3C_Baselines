import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        return True

class PPO_LSTM_Trainer:
    def __init__(self, env, seed=None):
        self.env = env
        self.seed = seed
        self.model = RecurrentPPO(
            "MlpLstmPolicy",      
            self.env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            policy_kwargs={
                "lstm_hidden_size": 32,
                "n_lstm_layers": 2,
                "enable_critic_lstm": True
            },
            clip_range=0.2,
            seed=seed,
        )
        
    def preprocess_state(self, state):
        """预处理观测状态"""
        # 如果state是numpy数组,确保其为float32类型
        if isinstance(state, np.ndarray):
            return state.astype(np.float32)
        # 如果是list,转换为numpy数组
        elif isinstance(state, list):
            return np.array(state, dtype=np.float32)
        # 其他情况直接返回
        return state
        
    def train(self, episodes=10, max_steps=1000):
        # 首先收集评估数据以便与训练后比较
        episode_count = 0
        total_steps = 0
        total_success = 0
        episode_returns = []
        
        # 执行评估回合
        while episode_count < min(3, episodes) and total_steps < min(max_steps // 3, 1000):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]    
            state = self.preprocess_state(state)
            done = False
            episode_reward = 0
            lstm_states = None
            
            while not done:
                try:
                    # 对于LSTM我们需要传递隐藏状态
                    action, lstm_states = self.model.predict(
                        state,
                        state=lstm_states,
                        deterministic=False
                    )
                    step_result = self.env.step(action)
                    if len(step_result) == 5:
                        next_state, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        next_state, reward, done, info = step_result
                    next_state = self.preprocess_state(next_state)
                    episode_reward += reward
                    total_steps += 1
                    state = next_state
                except Exception as e:
                    print(f"Error during evaluation step for PPO_LSTM: {e}")
                    done = True
                    break
            
            success = done and reward > 0
            total_success += int(success)
            episode_returns.append(episode_reward)
            episode_count += 1
        
        # 使用learn方法进行训练
        try:
            # 确保不超过最大步数
            total_timesteps = min(max_steps, 2048)  # 使用model.n_steps作为默认值
            
            # 创建回调函数以便在训练过程中进行监控
            callback = CustomCallback()
            # 使用learn方法训练模型
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                progress_bar=False
            )
            
            print(f"PPO_LSTM training completed with {total_timesteps} timesteps")
            
        except Exception as e:
            print(f"Error during PPO_LSTM training: {e}")
            import traceback
            traceback.print_exc()
        
        # 训练后再进行一次评估
        post_train_episode_count = 0
        post_train_total_steps = 0
        post_train_total_success = 0
        post_train_episode_returns = []
        
        while post_train_episode_count < min(episodes - episode_count, 5) and post_train_total_steps < min(max_steps - total_steps, 1000):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]
            state = self.preprocess_state(state)
            done = False
            episode_reward = 0
            lstm_states = None
            
            while not done:
                try:
                    action, lstm_states = self.model.predict(
                        state,
                        state=lstm_states,
                        deterministic=False
                    )
                    step_result = self.env.step(action)
                    if len(step_result) == 5:
                        next_state, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        next_state, reward, done, info = step_result
                    next_state = self.preprocess_state(next_state)
                    episode_reward += reward
                    post_train_total_steps += 1
                    state = next_state
                except Exception as e:
                    print(f"Error during post-training evaluation for PPO_LSTM: {e}")
                    done = True
                    break
            
            success = done and reward > 0
            post_train_total_success += int(success)
            post_train_episode_returns.append(episode_reward)
            post_train_episode_count += 1
        
        # 合并所有统计数据
        total_episode_count = episode_count + post_train_episode_count
        all_episode_returns = episode_returns + post_train_episode_returns
        total_success_count = total_success + post_train_total_success
        total_step_count = total_steps + post_train_total_steps + total_timesteps
        
        # 计算并返回结果
        avg_return = np.mean(all_episode_returns) if all_episode_returns else 0
        success_rate = total_success_count / total_episode_count if total_episode_count > 0 else 0
        
        print(f"PPO_LSTM - episodes={total_episode_count}, steps={total_step_count}, "
              f"success_rate={success_rate:.2%}, avg_return={avg_return:.2f}")
        
        return {
            "avg_return": avg_return,
            "episode_count": total_episode_count,
            "total_steps": total_step_count,
            "total_success": total_success_count,
            "episode_returns": all_episode_returns
        }
        
    def get_state_dict(self):
        """返回策略的状态字典"""
        return self.model.policy.state_dict().copy()