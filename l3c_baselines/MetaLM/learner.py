import sys
import numpy
import paddle.distributed.fleet as fleet
import paddle
import parl
import argparse
import os
import gym
import pickle
from .utils import metalm_loss_func, autoregressive_mask
from .utils import detached_memory, linear_segments, EpochStat, formulate_numpyarray
from .utils import load_model, save_model
from .actor import ActorMazeWorld
from .plasmer_block import PlasmerEncoderLayer 
from .plasmer_classifier import PlasmerClassifier
from .lstm_classifier import LSTMClassifier
from .transformer_classifier import TransformerClassifier
from .metagym.metalm import MetaMaze, MazeTaskSampler

class Learner(object):
    def __init__(self, server_ip, configs):
        parl.connect(server_ip, distributed_files=['./l3c_baselines/*.py', './*.py'])
        self._model = model
        self._actors = [ActorMazeWorld(configs)]
        self._inactive_actors = dict()
        self._max_steps = configs["inner_horizon"]
        self._eval_tasks_file = configs["eval_tasks_file"]
        if(self._eval_tasks_file is not None):
            self._eval_tasks = pickle.load(self._eval_tasks_file)
        else:
            self._eval_tasks = None

    def get_distributed_interactions(self, tasks=None):
        weights = self._model.get_weights()
        # Remote tasks recorder
        tasks = list()
        # Active tasks dict, if True, it is active
        active_task = dict()
        n_active_tasks = 0
        data = {"states": list(), "actions": list(), "rewards": list()}

        # Distribute Tasks and Run
        for i, actor in enumerate(self._actors):
            if(i not in self._inactive_actors):
                tasks.append(actor.exec_episodes(self._max_steps, weights))
                active_task[i] = True
            else:
                # In case node is no longer active, turn it off
                tasks.append(None)
                active_task[i] = False
            n_active_tasks += 1

        # Retrieve data
        wait_time = 0
        while wait_time < self._max_wait_time and n_active_tasks > 0:
            time.sleep(1)
            for key in active_task:
                if(active_task[key]):
                    try:
                        state_list, action_list, reward_list = tasks[key].get_nowait()
                        data["states"].append(state_list)
                        data["actions"].append(action_list)
                        data["rewards"].append(reward_list)
                        active_task[key] = False
                        n_active_tasks -= 1
                    except Exception:
                        pass
            wait_time += 1

        # Set the inactive node (Timeout)
        if(n_active_tasks > 0):
            for key in active_task:
                self._inactive_actors.add(i)
        return data
        
    def train(self):
        data = self.get_distributed_actions()
        self._model.train(data)

    def eval(data):
        if(self.eval_tasks is not None):
            data = self.get_distributed_actions(tasks=self.eval_tasks)
        avg_reward = numpy.mean(map(sum, data["rewards"]))
        return avg_reward

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
