import numpy
import torch

def img_pro(observations):
    return observations / 255

def img_post(observations):
    return observations * 255


def rewards2go(rewards, gamma=0.98):
    """
    returns a future moving average of rewards
    """
    rolled_rewards = rewards.clone()
    r2go = rewards.clone()
    n = max(min(50, -1/numpy.log10(gamma)), 0)
    for _ in range(n):
        rolled_rewards = gamma * torch.roll(rolled_rewards, shifts=-1, dims=1)
        r2go += rolled_rewards
    return r2go

def downsample(x, downsample_length, axis=-1):
    """
    Downsample and get the mean of each segment along a given axis
    """
    if(isinstance(x, torch.Tensor)):
        shape = x.shape
        ndim = x.dim()
    else:
        shape = numpy.shape(x)
        ndim = numpy.asarray(x).ndim
    if(axis < 0):
        axis = axis + ndim
    full_len = shape[axis]
    if(downsample_length >= full_len):
        if(isinstance(x, torch.Tensor)):
            return torch.mean(x, dim=axis, keepdim=True)
        else:
            return numpy.mean(x, axis=axis, keepdims=True)
    trunc_seg = full_len // downsample_length
    trunc_len = trunc_seg * downsample_length

    new_shape = shape[:axis] + (trunc_seg, downsample_length) + shape[axis + 1:]
    # Only when left elements is large enough we add the last statistics
    need_addition = (trunc_len + downsample_length // 2 < full_len)

    if(isinstance(x, torch.Tensor)):
        ds_x = torch.mean(torch.narrow(x, axis, 0, trunc_len).view(new_shape), dim=axis + 1, keepdim=False)
        if(need_addition):
            add_x = torch.mean(torch.narrow(x, axis, trunc_len, full_len - trunc_len), dim=axis, keepdim=True)
            ds_x = torch.cat((ds_x, add_x), dim=axis)
    else:
        slc = [slice(None)] * len(shape)
        slc[axis] = slice(0, trunc_len)
        x = numpy.array(x)
        reshape_x = numpy.reshape(x[tuple(slc)], new_shape)
        ds_x = numpy.mean(reshape_x, axis=axis + 1, keepdims=False)
        if(need_addition):
            slc[axis] = slice(trunc_len, full_len)
            add_x = numpy.mean(x[tuple(slc)], axis=axis, keepdims=True)
            ds_x = numpy.concatenate((ds_x, add_x), axis=axis)
    return ds_x

class MapStateToDiscrete:
    def __init__(self, env_name):
        self.env_name = env_name.lower()
        
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
        n_interval_theta = 8
        
        # Use the helper function to map theta
        theta_discrete = self.map_to_discrete(theta, theta_min, theta_max, n_interval_theta)
        
        # Define the range and number of intervals for speed
        speed_min, speed_max = -8.0, 8.0
        n_interval_speed = 8
        
        # Use the helper function to map speed
        speed_discrete = self.map_to_discrete(state[2], speed_min, speed_max, n_interval_speed)
        
        # Calculate the discretized state
        state_discrete = n_interval_theta * theta_discrete + speed_discrete
        
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
        n_interval_position = 8
        
        velocity_min, velocity_max = -0.07, 0.07
        n_interval_velocity = 8
        
        # Use the helper function to map position and velocity
        position_discrete = self.map_to_discrete(state[0], position_min, position_max, n_interval_position)
        velocity_discrete = self.map_to_discrete(state[1], velocity_min, velocity_max, n_interval_velocity)
        
        # Calculate the discretized state
        state_discrete = 8 * position_discrete + velocity_discrete
        
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
            raise ValueError(f"Unsupported environment: {env_name}")
    
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
        
        return continuous_action  
    
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