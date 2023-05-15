import sys
import numpy
import paddle.distributed.fleet as fleet
import paddle
import parl
import argparse
import os
import gym
from l3c.metamaze import MetaMaze, MazeTaskSampler
from .utils import metalm_loss_func, autoregressive_mask
from .utils import detached_memory, linear_segments, EpochStat, formulate_numpyarray
from .utils import load_model, save_model
from .plasmer_block import PlasmerEncoderLayer 

@parl.remote_class
class Actor(object):
    def __init__(self, config_dict, fixed_task=None):
        self._env = gym.make("meta-maze-discrete-3D-v0", enable_render="False", task_type="SURVIVAL")
        self._config_dict = config_dict
        #self._model = Need To Rewrite Here
        self._default_reward = 0
        self._default_action = -1
        if(fixed_task is not None):
            self.task = fixed_task
            self._env.set_task(self.task)
        else:
            self.task = sample_task()
            self._env.set_task(self.task)
        
    def sample_task(self):
        return MazeTaskSampler(
            n            = 15,  # Number of cells = n*n
            allow_loops  = False,  # Whether loops are allowed
            crowd_ratio  = 0.40,   # Specifying how crowded is the wall in the region, only valid when loops are allowed. E.g. crowd_ratio=0 means no wall in the maze (except the boundary)
            cell_size    = 2.0, # specifying the size of each cell, only valid for 3D mazes
            wall_height  = 3.2, # specifying the height of the wall, only valid for 3D mazes
            agent_height = 1.6, # specifying the height of the agent, only valid for 3D mazes
            view_grid    = 1, # specifiying the observation region for the agent, only valid for 2D mazes
            step_reward  = -0.01, # specifying punishment in each step in ESCAPE mode, also the reduction of life in each step in SURVIVAL mode
            goal_reward  = 1.0, # specifying reward of reaching the goal, only valid in ESCAPE mode
            initial_life = 1.0, # specifying the initial life of the agent, only valid in SURVIVAL mode
            max_life     = 2.0, # specifying the maximum life of the agent, acquiring food beyond max_life will not lead to growth in life. Only valid in SURVIVAL mode
            food_density = 0.01,# specifying the density of food spot in the maze, only valid in SURVIVAL mode
            food_interval= 100, # specifying the food refreshing periodicity, only valid in SURVIVAL mode
            )

    def exec_episode(self, max_steps, model_weights, resample=True):
        steps = 0
        if(resample):
            self.task = sample_task()
            self._env.set_task(self.task)
        init_state = self._env.reset()
        self._model.load_weights(model_weights)
        state_list = [init_state]
        action_list = [self._default_action]
        reward_list = [self._default_reward]
        while steps < max_steps:
            steps += 1
            action = self.model(state_list, action_list, reward_list)
            observation, reward, done, info = self._env.step(action)
            state_list.append(observation)
            action_list.append(action)
            reward_list.append(reward)
            if(done):
                state_list.append(self._env.reset())
                action_list.append(self._default_action)
                reward_list.append(self._default_reward)
        return state_list, action_list, reward_list

if __name__=="__main__":
    fleet.init(is_collective=True)

    tied_nlayers = 1
    max_len = 4096
    mode = "TransformerXL"
    #mode = "LSTM"
    train_segment = 128
    eval_segment = 128
    V = 512
    n = 8
    l = 64
    e = 0.10

    worker_num = fleet.worker_num()
    worker_index = fleet.worker_index()

    #fm = FileManager(sys.argv[1], worker_num, worker_index)
    DataGen = MetaLM(V, n, l, e, max_len)
    pad_id = DataGen.PaddingID
    vocab_size = DataGen.VocabSize

    if(worker_index == 0):
        eval_loader = FileDataset(sys.argv[1], pad_id)

    if(mode == "LSTM"):
        BATCH_SIZE = 256
        model = LSTMClassifier(512, vocab_size, dropout=0.10, nlayers=4, tied_nlayers=tied_nlayers)
    elif(mode == "TransformerXL"):
        BATCH_SIZE = 128
        model = TransformerClassifier(512, vocab_size, 6, 8)
    elif(mode == "Plasmer"):
        model = PlasmerClassifier(
            d_model, 
            n_head, 
            d_model *4,
            n_layer,
            pad_id
            )
        BATCH_SIZE = 128

    for name, parameter in model.named_parameters():
        print(name, parameter.shape)

    model = paddle.DataParallel(model)

    #lr = paddle.optimizer.lr.NoamDecay(d_model=1.0e-3, warmup_steps=1000)
    #lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=1.0e-3, T_max=1000, eta_min=1.0e-4)
    #lr = paddle.optimizer.lr.CyclicLR(base_learning_rate=2.0e-3, max_learning_rate=5.0e-5, step_size_up=10)
    #lr = paddle.optimizer.lr.InverseTimeDecay(learning_rate=5.0e-3, gamma=0.25, last_epoch=-1)
    lr = paddle.optimizer.lr.ExponentialDecay(learning_rate=1.0e-3, gamma=0.99)
    adam = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters(),
            grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
            )
    adam = fleet.distributed_optimizer(adam)

    #load model
    if(mode == "LSTM"):
        #load_model(model, adam, "./checkpoint.LSTM/epoch-82")
        pass
    elif(mode == "TransformerXL"):
        #load_model(model, adam, "./checkpoint.TransformerXL/epoch-53")
        pass
    elif(mode == "Plasmer"):
        pass


    #Add inner loss weights
    inner_loss_weights = linear_segments(max_len, warmup_steps=l, intermediate_steps=l)

    epoch = 0

    while epoch < MAX_EPOCH:

        batch_idx = 0
        epoch += 1
        epoch_loss = None
        epoch_toks = None
        # Training
        is_end = False
        train_stat = EpochStat()
        while batch_idx < MAX_ITERATION and not is_end:
            model.train()
            batch_idx += 1
            batch_loss, batch_toks, is_end = train_batch(
                    DataGen, 
                    model, 
                    pad_id,
                    opt=adam,
                    loss_weights=inner_loss_weights,
                    seg_size=train_segment,
                    detach_segments=1,
                    is_train=True
                    )
            train_stat.add_batch(batch_toks, batch_loss, None)
            print("Epoch: %d, batch %d, learning rate: %f, average loss: %s"%(epoch, batch_idx, adam.get_lr(), formulate_numpyarray(batch_loss / batch_toks)))
            sys.stdout.flush()
        lr.step()
        epoch_loss, _ = train_stat.get_statistics()
        print("[TRAINING] Epoch: %d, training loss: %s"%(epoch, formulate_numpyarray(epoch_loss)))
        sys.stdout.flush()

        if(worker_index == 0):
            save_model(model, adam, "checkpoint.%s/epoch"%mode, epoch)
        
        # Testing
        if(worker_index == 0):
            eval_loader.reset(BATCH_SIZE)
            is_end = False
            epoch_loss = None
            epoch_toks = None
            model.eval()
            eval_stat = EpochStat()
            while not is_end:
                batch_loss, batch_toks, is_end = train_batch(
                        eval_loader, 
                        model, 
                        pad_id,
                        seg_size=eval_segment,
                        is_train=False
                        )
                eval_stat.add_batch(batch_toks, batch_loss, None)
            epoch_loss, _ = eval_stat.get_statistics()
            print("[EVALUATING] Epoch: %d, evaluation ppl: %s"%(epoch, formulate_numpyarray(epoch_loss)))
            sys.stdout.flush()
