import sys
import numpy
import paddle.distributed.fleet as fleet
import paddle
import argparse
import os
import gym
from plasmer_block import PlasmerEncoderLayer 
from utils import metalm_loss_func, autoregressive_mask, detached_memory, linear_segments, EpochStat, formulate_numpyarray
from plasmer_classifier import PlasmerClassifier
from lstm_classifier import LSTMClassifier
from transformer_classifier import TransformerClassifier
from metagym.metalm import MetaMaze, MazeTaskSampler


MAX_EPOCH = 100
MAX_ITERATION = 100

if os.name == 'nt':
    import win32api, win32con
def file_is_hidden(p):
    if os.name== 'nt':
        attribute = win32api.GetFileAttributes(p)
        return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
    else:
        return p.startswith('.') #linux-osx

class Actor(object):
    def __init__(self, model):
		self._env = gym.make("meta-maze-discrete-3D-v0", enable_render="False", task_type="SURVIVAL")
		
	def sample_task(self):
        self.task = MazeTaskSampler(
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
       	self._env.set_task(self.task)
        self._env.reset()

	def post_process_data(self):

	def exec_episode(self, max_steps):

def debug_print_grad(model):
    for name, parameter in model.named_parameters():
        print(name, paddle.linalg.norm(parameter.grad, p=numpy.inf))

def debug_print_para(model):
    for name, parameter in model.named_parameters():
        print("parameter", name, paddle.linalg.norm(parameter, p=numpy.inf))

def debug_print_norm(mems):
    for mem in mems[0]:
        print("mem", paddle.linalg.norm(mem, p=numpy.inf))

def train_batch(data_generator, model, pad_id,
        loss_weights=None,
        seg_size=256, 
        opt=None, 
        detach_segments=4, 
        is_train=True):

    if(opt is None and is_train):
        raise Exception("Must set opt when is_train == True")
    if(isinstance(data_generator, FileDataset)):
        feas, labs, is_end = data_generator.batch()
    elif(isinstance(data_generator, MetaLM)):
        feas, labs = data_generator.batch_generator(BATCH_SIZE)
        is_end = False
    elif(isinstance(data_generator, list)):
        feas, labs = data_generator
        is_end = False
    else:
        raise Exception("Unknown data_generator type:", data_generator)
    mem = None
    acc_loss = []
    batch_ppl = []
    batch_toks = []
    seg_idx = 0
    for b_idx, e_idx, fea, lab, seg_toks in INNER_ITERATOR(feas, labs, seg_size, pad_id):
        seg_idx += 1
        src_mask = autoregressive_mask(lab)
        (loss, mem) = metalm_loss_func(model,
                paddle.to_tensor(fea, dtype="int32"), 
                paddle.to_tensor(lab, dtype="int32"), 
                src_mask=src_mask,
                mems=mem)
        #debug_print_norm(mem)
        ppl = paddle.sum(loss)
        batch_ppl.append(ppl.numpy()[0])
        batch_toks.append(seg_toks)
        if(is_train):
            if(loss_weights is None):
                acc_loss.append([seg_toks, paddle.sum(loss)])
            else:
                acc_loss.append([seg_toks, paddle.sum(loss * loss_weights[b_idx:e_idx])])
            if(seg_idx % detach_segments == 0):
                sum_loss = 0
                sum_toks = 0

                for toks, loss in acc_loss:
                    sum_loss += loss
                    sum_toks += toks
                sum_loss = (1.0 / (max(sum_toks, 1.0))) * sum_loss

                sum_loss.backward()
                #debug_print_grad(model)
                opt.step()
                opt.clear_grad()
                acc_loss = []
                mem = detached_memory(mem)
        else:
            mem = detached_memory(mem)

    #print("Memory allocated:", paddle.device.cuda.memory_allocated())
    if(len(acc_loss) > 0):
        sum_loss = 0
        sum_toks = 0

        for toks, loss in acc_loss:
            sum_loss += loss
            sum_toks += toks
        sum_loss = (1.0 / (max(sum_toks, 1.0))) * sum_loss

        sum_loss.backward()
        #debug_print_grad(model)
        opt.step()
        opt.clear_grad()

    acc_loss = []
    mem = detached_memory(mem)
    #debug_print_para(model)

    return numpy.asarray(batch_ppl), numpy.asarray(batch_toks), is_end

def load_model(model, opt, file_name, load_opt=False):
    layer_state_dict = paddle.load("%s.pdparams"%file_name)
    model.set_state_dict(layer_state_dict)
    if(load_opt):
        opt_state_dict = paddle.load("%s.pdopt"%file_name)
        opt.set_state_dict(opt_state_dict)

def save_model(model, opt, file_prefix, epoch):
    save_file_name = "%s-%d"%(file_prefix, epoch)
    print("Saving models to %s"%(save_file_name))
    paddle.save(model.state_dict(), save_file_name + ".pdparams")
    paddle.save(opt.state_dict(), save_file_name + ".pdopt")


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
