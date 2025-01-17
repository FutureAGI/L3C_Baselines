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