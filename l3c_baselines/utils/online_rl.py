import numpy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import A2C, PPO, DQN, TD3
import gym

class MapStateToDiscrete:
    def __init__(self, env_name, state_space_dim1, state_space_dim2):
        self.env_name = env_name.lower()
        self.state_space_dim1 = state_space_dim1
        self.state_space_dim2 = state_space_dim2
        
        if self.env_name.find("pendulum") >= 0:
            self.map_state_to_discrete_func = self._map_state_to_discrete_pendulum
        elif self.env_name.find("mountaincar") >= 0:
            self.map_state_to_discrete_func = self._map_state_to_discrete_mountaincar
        else:
            self.map_state_to_discrete_func = self._map_state_to_discrete_default # return origin state
            raise ValueError(f"Unsupported environment: {env_name}")
    
    def map_to_discrete(self, value, min_val, max_val, n_interval):
        """
        Maps a continuous value to a discrete integer.

        Parameters:
        value (float): The continuous value to be mapped.
        min_val (float): The minimum value of the continuous range.
        max_val (float): The maximum value of the continuous range.
        n_interval (int): The number of intervals.

        Returns:
        int: The mapped discrete integer [0, n_interval - 1].
        """
        # Create bin edges
        bins = numpy.linspace(min_val, max_val, n_interval + 1)
        
        # Clip the value within the range [min_val, max_val]
        clipped_value = numpy.clip(value, min_val, max_val)
        
        # Digitize the clipped value to get the discrete integer
        discrete_value = numpy.digitize(clipped_value, bins) - 1
        
        # Ensure the discrete value is within the range [0, num_bins-1]
        return numpy.clip(discrete_value, 0, n_interval - 1)
    
    def _map_state_to_discrete_pendulum(self, state):
        """
        Maps a state array to a discrete integer for Pendulum-v1.

        Parameters:
        state (numpy.ndarray): An array containing cos(theta), sin(theta), and speed.

        Returns:
        int: The discretized state integer.
        """
        # Extract cos_theta and sin_theta
        cos_theta = state[0]
        sin_theta = state[1]
        
        # Calculate theta using atan2 to get the correct quadrant
        theta = numpy.arctan2(sin_theta, cos_theta)
        
        # Map theta from [-pi, pi] to [0, 2*pi]
        if theta < 0:
            theta += 2 * numpy.pi
        
        # Define the range and number of intervals for theta
        theta_min, theta_max = 0, 2 * numpy.pi
        n_interval_theta = self.state_space_dim1
        
        # Use the helper function to map theta
        theta_discrete = self.map_to_discrete(theta, theta_min, theta_max, n_interval_theta)
        
        # Define the range and number of intervals for speed
        speed_min, speed_max = -8.0, 8.0
        n_interval_speed = self.state_space_dim2
        
        # Use the helper function to map speed
        speed_discrete = self.map_to_discrete(state[2], speed_min, speed_max, n_interval_speed)
        
        # Calculate the discretized state
        state_discrete = n_interval_speed * theta_discrete + speed_discrete
        
        return state_discrete
    
    def _map_state_to_discrete_mountaincar(self, state):
        """
        Maps a state array to a discrete integer for MountainCar-v0.

        Parameters:
        state (numpy.ndarray): An array containing position and velocity.

        Returns:
        int: The discretized state integer.
        """
        # Define the ranges and number of intervals for position and velocity
        position_min, position_max = -1.2, 0.6
        n_interval_position = self.state_space_dim1
        
        velocity_min, velocity_max = -0.07, 0.07
        n_interval_velocity = self.state_space_dim2
        
        # Use the helper function to map position and velocity
        position_discrete = self.map_to_discrete(state[0], position_min, position_max, n_interval_position)
        velocity_discrete = self.map_to_discrete(state[1], velocity_min, velocity_max, n_interval_velocity)
        
        # Calculate the discretized state
        state_discrete = n_interval_velocity * position_discrete + velocity_discrete
        
        return state_discrete
    
    def _map_state_to_discrete_default(self, state):
        return state
    
    def map_state_to_discrete(self, state):
        """
        Maps a state array to a discrete integer based on the environment.

        Parameters:
        state (numpy.ndarray): An array containing the state variables of the environment.

        Returns:
        int: The discretized state integer.
        """
        return self.map_state_to_discrete_func(state)
    
class MapActionToContinuous:
    def __init__(self, env_name, distribution_type='linear'):
        self.env_name = env_name.lower()
        self.distribution_type = distribution_type
        
        if self.env_name.find("pendulum") >= 0:
            self.map_action_to_continuous_func = self._map_action_to_continous_pendulum
        else:
            self.map_action_to_continuous_func = self._map_action_to_continous_default # return origin action
    
    def map_to_continuous(self, value, min_val, max_val, n_action):
        """
        Maps a discrete integer to a continuous value.

        Parameters:
        value (int): The discrete integer to be mapped.
        min_val (float): The minimum value of the continuous range.
        max_val (float): The maximum value of the continuous range.
        n_interval (int): The number of intervals.

        Returns:
        float: The mapped continuous value within the range [min_val, max_val].
        """
        # Calculate the step size for each interval
        if n_action < 2:
            raise ValueError(f"Invalid number of actions: {n_action}")
        
        if self.distribution_type == 'linear':
            step_size = (max_val - min_val) / (n_action - 1)
            continuous_value = min_val + value * step_size
        elif self.distribution_type == 'sin':
            # Map the discrete value to a normalized range [0, pi]
            normalized_value = (value / (n_action - 1)) * numpy.pi
            # Apply sine function and scale it to the desired range
            continuous_value = min_val + ((numpy.sin(normalized_value) + 1) / 2) * (max_val - min_val)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
        
        return continuous_value
    
    def _map_action_to_continous_pendulum(self, action):
        """
        Maps a discrete action integer to a continuous action for Pendulum-v1.

        Parameters:
        action (int): A discrete action integer from 0 to n_action-1.

        Returns:
        float: The mapped continuous action value.
        """
        min_val, max_val = -2.0, 2.0
        n_action = 5
        
        # Use the helper function to map action
        continuous_action = self.map_to_continuous(action, min_val, max_val, n_action)
        
        return numpy.array([continuous_action])  
    
    def _map_action_to_continous_default(self, action):
        return action
    
    def map_action_to_continuous(self, action):
        """
        Maps a discrete action integer to a continuous action based on the environment.

        Parameters:
        action (int): A discrete action integer from 0 to n_action-1.

        Returns:
        float: The mapped continuous action value.
        """
        return self.map_action_to_continuous_func(action)
    
class DiscreteEnvWrapper(gym.Wrapper):
    def __init__(self, env, env_name, action_space=5, state_space_dim1=8, state_space_dim2=8, reward_shaping = False, skip_frame=0):
        super(DiscreteEnvWrapper, self).__init__(env)
        self.env_name = env_name.lower()
        self.action_space = gym.spaces.Discrete(action_space)
        self.observation_space = gym.spaces.Discrete(state_space_dim1 * state_space_dim2)
        self.reward_shaping = reward_shaping
        self.skip_frame = skip_frame
        self.map_state_to_discrete = MapStateToDiscrete(self.env_name, state_space_dim1, state_space_dim2).map_state_to_discrete
        self.map_action_to_continuous = MapActionToContinuous(self.env_name).map_action_to_continuous

        self.last_speed = 0.0
    def reset(self, **kwargs):
        continuous_state, info = self.env.reset(**kwargs)
        discrete_state = self.map_state_to_discrete(continuous_state)
        return discrete_state, info
        
    def step(self, discrete_action):
        total_reward = 0.0
        continuous_action = self.map_action_to_continuous(discrete_action)
        for _ in range(self.skip_frame + 1):
            continuous_state, reward, terminated, truncated, info = self.env.step(continuous_action)
            if self.reward_shaping:
                if self.env_name.lower().find("mountaincar") >= 0:
                    reward = 0.1*reward + 10 * numpy.abs(continuous_state[1] - self.last_speed)
                    self.last_speed = continuous_state[1]
            total_reward += reward
            if terminated or truncated:
                break
        discrete_state = self.map_state_to_discrete(continuous_state)
        return discrete_state, total_reward, terminated, truncated, info

    def render(self):
        return self.env.render()
    def close(self):
        return self.env.close()

class RolloutLogger(BaseCallback):
    """
    A custom callback for logging the total reward and episode length of each rollout.
    
    :param env_name: Name of the environment.
    :param max_rollout: Maximum number of rollouts to perform.
    :param max_step: Maximum steps per episode.
    :param downsample_trail: Downsample trail parameter.
    :param verbose: Verbosity level: 0 = no output, 1 = info, 2 = debug
    """
    def __init__(self, env_name, max_rollout, max_step, downsample_trail, verbose=0):
        super(RolloutLogger, self).__init__(verbose)
        self.env_name = env_name.lower()
        self.max_rollout = max_rollout
        self.max_steps = max_step
        self.current_rollout = 0
        self.reward_sums = []
        self.step_counts = []
        self.success_rate = []
        self.success_rate_f = 0.0
        self.downsample_trail = downsample_trail
        self.episode_reward = 0
        self.episode_length = 0

    def is_success_fail(self, reward, total_reward, step):
        if "lake" in self.env_name:
            return int(reward > 1.0e-3)
        elif "lander" in self.env_name:
            return int(total_reward >= 200)
        elif "mountaincar" in self.env_name:
            return int(step < self.max_steps)
        else:
            return 0

    def _on_step(self) -> bool:
        """
        This method is called after every step in the environment.
        Here we update the current episode's reward and length.
        """
        # Accumulate the episode reward
        self.episode_reward += self.locals['rewards'][0]
        self.episode_length += 1

        if self.locals['dones'][0]:
            # Episode is done, record the episode information
            succ_fail = self.is_success_fail(self.locals['rewards'][0], self.episode_reward, self.episode_length)
            
            if self.current_rollout < self.downsample_trail:
                self.success_rate_f = (1 - 1 / (self.current_rollout + 1)) * self.success_rate_f + succ_fail / (self.current_rollout + 1)
            else:
                self.success_rate_f = (1 - 1 / self.downsample_trail) * self.success_rate_f + succ_fail / self.downsample_trail

            self.reward_sums.append(self.episode_reward)
            self.step_counts.append(self.episode_length)
            self.success_rate.append(self.success_rate_f)

            # Reset episode counters
            self.episode_reward = 0
            self.episode_length = 0

            # Check if we have reached the maximum number of rollouts
            self.current_rollout += 1
            if self.current_rollout >= self.max_rollout:
                if self.verbose >= 1:
                    print(f"Reached maximum rollouts ({self.max_rollout}). Stopping training.")
                self.model.stop_training = True
                return False

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        For algorithms that do not use rollout_buffer, this method can be left empty.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered at the end of training.
        We can perform any final logging here if needed.
        """
        # If there are remaining episode data, log it
        if self.episode_length > 0:
            succ_fail = self.is_success_fail(self.episode_reward, self.episode_reward, self.episode_length)
            
            if self.current_rollout < self.downsample_trail:
                self.success_rate_f = (1 - 1 / (self.current_rollout + 1)) * self.success_rate_f + succ_fail / (self.current_rollout + 1)
            else:
                self.success_rate_f = (1 - 1 / self.downsample_trail) * self.success_rate_f + succ_fail / self.downsample_trail

            self.reward_sums.append(self.episode_reward)
            self.step_counts.append(self.episode_length)
            self.success_rate.append(self.success_rate_f)


class OnlineRL:
    def __init__(self, env, env_name, model_name, max_trails, max_steps, downsample_trail):
        self.env = env
        self.model_name = model_name
        self.log_callback = RolloutLogger(env_name, max_trails, max_steps, downsample_trail, verbose=1)
        
    def create_model(self):
        model_classes = {'a2c': A2C, 'ppo': PPO, 'dqn': DQN, 'td3': TD3}
        if self.model_name not in model_classes:
            raise ValueError("Unknown policy type: {}".format(self.model_name))
        
        # Create the model with appropriate parameters
        if self.model_name.lower() in ['a2c', 'ppo']:
            self.model = model_classes[self.model_name.lower()](
                policy='MlpPolicy', env=self.env, verbose=1)
        elif self.model_name.lower() == 'dqn':
            self.model = DQN(
                policy='MlpPolicy', env=self.env,
                learning_rate=0.00025, buffer_size=100_000, exploration_fraction=0.1,
                exploration_final_eps=0.01, batch_size=32, tau=0.005,
                train_freq=(4, 'step'), gradient_steps=1, seed=None, optimize_memory_usage=False,
                verbose=1)
        elif self.model_name.lower() == 'td3':
            self.model = TD3(
                policy='MlpPolicy', env=self.env,
                learning_rate=0.0025, buffer_size=1_000_000, train_freq=(1, 'episode'),
                gradient_steps=1, action_noise=None, optimize_memory_usage=False,
                replay_buffer_class=None, replay_buffer_kwargs=None, verbose=1)

    def __call__(self):
        self.create_model()
        self.model.learn(total_timesteps=int(1e6), callback=self.log_callback)
        return (self.log_callback.reward_sums, 
                self.log_callback.step_counts, 
                self.log_callback.success_rate)

if __name__ == "__main__":
    model_name = "dqn"
    env_name = "lake"
    max_trails = 50
    max_steps = 200
    downsample_trail = 10

    if env_name == "lake":
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    else:
        raise ValueError(f"Unknown environment: {env_name}")

    online_rl = OnlineRL(env, env_name, model_name, max_trails, max_steps, downsample_trail)
    reward_sums, step_counts, success_rate = online_rl()
    
    print("Reward Sums:", reward_sums)
    print("Step Counts:", step_counts)
    print("Success Rate:", success_rate)