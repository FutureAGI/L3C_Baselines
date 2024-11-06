# L3C_Baselines
Implementations of baselines for [L3C](https://github.com/FutureAGI/L3C)

# Motivations
Generative models presents promising capability to generalize and adapt to different tasks. The so-called "In-Context Learning" (ICL) can efficiently adapt to new tasks without tuning the parameters with high sample efficiency. We try build foundation models for learning to learn with ICL, covering the domain of language modeling, world modeling, and decision modeling.

# Directory Structure
- `[projects](./projects)`: implementations of model training and validating for different projects in L3C.
    - `[MetaLM](./projects/MetaLM)` foundation model for [L3C MetaLM](https://github.com/FutureAGI/L3C/tree/main/l3c/metalang)
    - `[MazeWorld](./projects/MazeWorld）` foundation model for [L3C MazeWorld](https://github.com/FutureAGI/L3C/tree/main/l3c/mazeworld)

- `data`: For general-purpose learning to learn, we generate the datasets by procedurally sampling tasks and use expert's demonstration for imitation learning and self-supervised learning. This directory contains the scripts to generate unlimited datasets for training.

- `l3c_baselines`: contains the building blocks and utils of different models
    - `modules`: contains the basic blocks
    - `utils`: contains the utils for building networks, training, and evaluation
    - `models`: contains higher-level models built from basic blocks
    - `dataloader`: contains the dataloader for different tasks


# Training and Evaluating

## Install Requirements
To train a model run
```bash
git clone https://github.com/FutureAGI/L3C_Baselines
cd L3C_Baselines
pip install -e .
```

## Generate Datasets

check the [data](./data) directory to generate datasets for training and evaluation.

## Start Training

### Reconfigure the Config File

Basically you need to modify the configuration file to start the training. The config file basically need to contain three major parts:
- `log_config`: configuration of the log directory, including the path to save the checkpoints and tensorboard logs
- `model_config`: configuration of the model structure and hyperparameters
- `train_config`: configuration of the training process, including learning rate, batch size, etc.
- `test_config`: configuration of the evaluation process, including the dataset to be evaluated

### Start Training

To train a model run
```bash
cd L3C_Baselines/projects/xxx
python train.py config.yaml
```

You might also overwrite the config file with command line arguments with ```--config```
```bash
python train.py config.yaml --configs key1=value1 key2=value2 ...
```

### Validating with static dataset
```bash
python validate.py config.yaml --configs key1=value1 key2=value2 ...
```

### Validating with interaction
Under development.

Feel free to submit a pull request.