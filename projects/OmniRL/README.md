OmniRL (Omnipotent-Reinforcement-Leanring) is an in-context reinforcement learning framework meta-trained on the large-scale dataset of Markov Decision Processes (MDPs).

## Features of OmniRL
- **Generalized In-context Learning**: OmniRL can learn a novel MDP task by in-context learning using imitation learning, reinforce learning, and offline-RL.
- **Long Trajectory Learning**: Can reasoning over trajectories as long as 1 million steps.
- **Generalization**: OmniRL can generalize to unseen MDPs and environments, including Cliff, Lake, MountainCar, Pendulum etc.

## Datasets and Models

You may download the datasets and models from [here]().

## Performances

A sketch of the structure of OmniRL
<div style="height: 320; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/OmniRL_Structure.gif" alt="OmniRL Sketch" style="width: 320px;">
</div>

OmniRL is completely meta-trained on synthetic MDPs ([AnyMDP](https://github.com/FutureAGI/L3C/tree/main/l3c/anymdp)) and can generalize to various environments.
<div style="height: 320; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/AnyMDP_Visualization.png" alt="OmniRL Train" style="width: 320px;">
</div>

Demonstration of OmniRL on various environments:
<div style="height: 320; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/OmniRLDemo1.gif" alt="OmniRL Demo" style="height: 100px;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/OmniRLDemo2.gif" alt="OmniRL Demo" style="height: 100px;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/OmniRLDemo3.gif" alt="OmniRL Demo" style="height: 100px;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/OmniRLDemo4.gif" alt="OmniRL Demo" style="height: 100px;">
</div>

Performance of OmniRL on Lake benchmarks and its capability to ICL from random trajectory:
<div style="height: 320; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/OmniRL_Figure.png" alt="OmniRL Performance" style="width: 320px;">
</div>

## Configuration

The `config.yaml` file contains all the necessary configuration for running OmniRL. Below is a detailed explanation of each field.

### General Configuration

Settings for general configuration.


### Log Configuration

Settings for logging and saving models.


### Model Configuration

Configuration for the overall model architecture and components, including encoders, decoders, and causal blocks. It defines the structure and behavior of the model during training and inference.

- **max_position**: Defines the maximum sequence length that the model can handle.

- **context_warmup**: The number of warm-up steps for the model at the beginning of training.

- **rsa_type**: Specifies the way the model processes the input data.If adding a new category is needed, modify `rsa_choice` field in `decision_model`

- **causal_block**: This module is used for handling causal relationships in the model. Setting `model_type`  is recommended for standard transformer architectures. Other options include `GSA`, `GLA`, `MAMBA`, and `RWKV6`, each offering different mechanisms for handling causal dependencies.Continue setting other model parameters after choosing `model_type`.

### Training Configuration

Settings for training the model.

- **seq_len**: Sequence length for training, better not less than `max_steps` while generating data.

### Test Configuration

Similar to the training setup above.

**Parameters not explicitly shown above may retain their default values as per the recommended configuration. Custom adjustments are available when aligned with specific application requirements.**