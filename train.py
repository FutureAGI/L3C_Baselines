import sys
import numpy
import paddle.distributed.fleet as fleet
import paddle
import argparse
import os
import paddle.profiler as profiler
import time
import l3c
from l3c_baselines.utils import metalm_loss_func
from l3c_baselines.utils import detached_memory, linear_segments, EpochStat, formulate_numpyarray
from l3c_baselines.utils import load_model, save_model
from l3c_baselines.utils import debug_print_norm, debug_print_grad, debug_print_para
from l3c_baselines.LSTM_block import AutoRegressiveLSTM
from l3c_baselines.RMT_block import AutoRegressiveRMT
from l3c_baselines.TRN_block import AutoRegressiveTRN
from l3c.metalm import MetaLMv1, MetaLMv2

MAX_EPOCH = 200
MAX_ITERATION = 100

if os.name == 'nt':
    import win32api, win32con
def file_is_hidden(p):
    if os.name== 'nt':
        attribute = win32api.GetFileAttributes(p)
        return attribute & (win32con.FILE_ATTRIBUTE_HIDDEN | win32con.FILE_ATTRIBUTE_SYSTEM)
    else:
        return p.startswith('.') #linux-osx

class FileManager(object):
    def __init__(self, file_dir, worker_num, worker_index):
        if(os.path.isfile(file_dir)):
            self.file_name = [file_dir]
            self.cur_index = 0
        else:
            self.file_name = [f for f in os.listdir(file_dir) if not file_is_hidden(f)]
            self.file_name = list(map(lambda x:os.path.join(file_dir, x), self.file_name))
            self.worker_num = worker_num
            self.cur_index = worker_index
        self.previous_file_name = None

    def get_file(self):
        file_name = self.file_name[self.cur_index]
        self.cur_index += 1
        self.cur_index = self.cur_index % len(self.file_name)
        if(self.previous_file_name == file_name):
            is_same = True
        else:
            is_same = False
        self.previous_file_name = file_name
        return file_name, is_same

class FileDataset(object):
    def __init__(self, file_name, pad_id):
        self.data_list = []
        self.data_len_list = []
        n_line = 0
        f_in = open(file_name, "r")
        for line in f_in:
            all_tokens = line.strip().split("\t")
            data = numpy.asarray(list(map(lambda x:list(map(int, x.split(","))), all_tokens)), dtype="int32")
            self.data_list.append(data)
            self.data_len_list.append(data.shape[0])
            n_line += 1
        f_in.close()
        self.sample_nums = n_line
        self.data_len_list = numpy.asarray(self.data_len_list, dtype="int32")
        self.pad_id = pad_id

    def reset(self, batch_size):
        self.shuf_idx = numpy.arange(self.sample_nums)
        numpy.random.shuffle(self.shuf_idx)
        self.cur_idx = 0
        self.batch_size = batch_size

    def batch(self):
        b_idx = self.cur_idx
        e_idx = min(self.batch_size + self.cur_idx, self.sample_nums)
        if(e_idx >= self.sample_nums):
            is_end = True
        else:
            is_end = False
        self.cur_idx = e_idx

        real_batch_size = e_idx - b_idx
        if(real_batch_size < 1):
            raise Exception("Iteration reaches the end, need reset before sampling batch")

        batch_idxes = self.shuf_idx[b_idx : e_idx]
        sel_lens = self.data_len_list[batch_idxes]

        # Must Synchronize Lengths across all workers
        max_len = numpy.max(sel_lens)

        features = numpy.full((real_batch_size, max_len), self.pad_id)
        labels = numpy.full((real_batch_size, max_len), self.pad_id)
        for i, idx in enumerate(batch_idxes.tolist()):
            features[i][:sel_lens[i]] = self.data_list[idx][:,0]
            labels[i][:sel_lens[i]] = self.data_list[idx][:,1]

        return features, labels, is_end

    def __getitem__(self, idx):
        return self.data_list[idx]

    def __len__(self):
        return self.n_line

def INNER_ITERATOR(fea, lab, segment_size, pad_id):
    b_idx = 0
    seg_idx = 0
    max_len = fea.shape[1]
    bsz = fea.shape[0]
    while b_idx < max_len:
        seg_idx += 1
        e_idx = min(b_idx + segment_size, max_len)
        if(e_idx <= b_idx):
            return
        seg_toks = (e_idx - b_idx) * bsz
        yield b_idx, e_idx, fea[:, b_idx:e_idx], lab[:, b_idx:e_idx], seg_toks
        b_idx = e_idx


def train_batch(
        data_generator, 
        model, 
        pad_id,
        worker_index,
        loss_weights=None,
        seg_size=256, 
        opt=None, 
        detach_segments=4, 
        batch_size=64,
        is_train=True):

    if(opt is None and is_train):
        raise Exception("Must set opt when is_train == True")
    if(isinstance(data_generator, FileDataset)):
        feas, labs, is_end = data_generator.batch()
    elif(isinstance(data_generator, MetaLMv1) or isinstance(data_generator, MetaLMv2)):
        seed = int(10 * time.time() + worker_index * 1000000) % 4294967295
        feas, labs = data_generator.batch_generator(batch_size, seed=seed)
        is_end = False
    elif(isinstance(data_generator, list)):
        feas, labs = data_generator
        is_end = False
    else:
        raise Exception("Unknown data_generator type:", data_generator)
    mem = None
    acc_loss = []
    batch_loss = []
    batch_ppl = []
    batch_toks = []
    seg_idx = 0

    #prof = profiler.Profiler(targets=[profiler.ProfilerTarget.GPU], 
    #        profile_memory=True, timer_only=True)
    for b_idx, e_idx, fea, lab, seg_toks in INNER_ITERATOR(feas, labs, seg_size, pad_id):
        seg_idx += 1
        #print("Worker index", worker_index, "Memory allocated before:", paddle.device.cuda.memory_allocated())
        (loss, mem) = metalm_loss_func(model,
                paddle.to_tensor(fea, dtype="int32"), 
                paddle.to_tensor(lab, dtype="int32"), 
                mems=mem)
        #print("loss.shape", loss.shape, "seg_toks", seg_toks, "from", b_idx, "to", e_idx)
        #debug_print_norm(mem)
        ppl = paddle.sum(loss)
        batch_ppl.append(ppl.numpy()[0])
        batch_toks.append(seg_toks)
        batch_loss.append(numpy.asarray(loss))
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

    if(len(acc_loss) > 0):
        sum_loss = 0
        sum_toks = 0

        for toks, loss in acc_loss:
            sum_loss += loss
            sum_toks += toks
        sum_loss = (1.0 / (max(sum_toks, 1.0))) * sum_loss

        sum_loss.backward()
        opt.step()
        opt.clear_grad()

    acc_loss = []
    pos_loss = numpy.mean(numpy.concatenate(batch_loss, axis=-1), axis=0)
    mem = detached_memory(mem)
    #debug_print_para(model)

    return numpy.asarray(batch_ppl), numpy.asarray(batch_toks), pos_loss, is_end

if __name__=="__main__":
    fleet.init(is_collective=True)

    parser = argparse.ArgumentParser(description='Training L3C Baselines')
    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--vocab_size', type=int, default=64)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--elements_length', type=int, default=64)
    parser.add_argument('--elements_number', type=int, default=10)
    parser.add_argument('--error_rate', type=float, default=0.05)
    parser.add_argument('--sequence_length', type=int, default=4096)
    parser.add_argument('--train_segment', type=int, default=128)
    parser.add_argument('--eval_segment', type=int, default=128)
    parser.add_argument('--model_type', type=str, default="LSTM")
    parser.add_argument('--model_layer_num', type=int, default=1)
    parser.add_argument('--model_hidden_size', type=int, default=512)
    parser.add_argument('--model_head_num', type=int, default=16)
    parser.add_argument('--model_memory_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--detach_segments', type=int, default=1)
    parser.add_argument('--opt_learning_rate', type=float, default=1.0e-4)
    parser.add_argument('--opt_warmup_steps', type=float, default=2000)
    parser.add_argument('--model_load_path', type=str, default=None)
    parser.add_argument('--train_warmup_steps', type=int, default=None)
    parser.add_argument('--train_intermediate_steps', type=int, default=None)
    parser.add_argument('--evaluation_data_path', type=str, default=None)

    args = parser.parse_args()

    train_segment = args.train_segment
    eval_segment = args.eval_segment
    max_segment = max(train_segment, eval_segment)
    V = args.vocab_size
    n = args.elements_number
    l = args.elements_length
    L = args.sequence_length
    e = args.error_rate
    nh = args.hidden_size
    ne = args.embedding_size
    version = args.version
    detach_segments = args.detach_segments
    
    model_type = args.model_type
    n_layer = args.model_layer_num
    n_hidden = args.model_hidden_size
    n_head = args.model_head_num
    n_mem = args.model_memory_size
    load_model_path = args.model_load_path
    eval_dataset = args.evaluation_data_path
    batch_size = args.batch_size
    opt_lr = args.opt_learning_rate
    opt_steps = args.opt_warmup_steps

    train_warmup_steps = args.train_warmup_steps
    train_intermediate_steps = args.train_intermediate_steps

    worker_num = fleet.worker_num()
    worker_index = fleet.worker_index()

    if(version == 'v1'):
        DataGen = MetaLMv1(V, n, l, e, L)
        pad_id = DataGen.PaddingID
        vocab_size = DataGen.VocabSize
    elif(version == 'v2'):
        DataGen = MetaLMv2(V, ne, nh, e, L)
        pad_id = DataGen.PaddingID
        vocab_size = DataGen.VocabSize
    elif(version == 'debug'):
        DataGen = FileDataset(eval_dataset, 0)
        pad_id = 0
        vocab_size = V
    else:
        raise Exception("no such version %s"%version)


    if(worker_index == 0):
        eval_loader = FileDataset(eval_dataset, pad_id)

    if(model_type == "LSTM"):
        model = AutoRegressiveLSTM(
            vocab_size, 
            n_hidden,
            n_layer)
    elif(model_type == "RMT"):
        model = AutoRegressiveRMT(
            vocab_size, 
            n_hidden, 
            n_layer, 
            n_head,
            seg_len=max_segment,
            mem_len=n_mem
            )
    elif(model_type == "TRN"):
        model = AutoRegressiveTRN(
            vocab_size, 
            n_hidden, 
            n_layer, 
            n_head,
            seg_len=max_segment,
            )
    else:
        raise Exception("No such model type: %s" % model_type)

    #model_info = paddle.summary(model, input_size=(1, 256), dtypes="int32", input=None)
    #print(model_info)
    #debug_print_para(model)

    model = paddle.DataParallel(model)

    lr = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=opt_lr, T_max=10, eta_min=1.0e-3 * opt_lr, verbose=True)
    #lr = paddle.optimizer.lr.CyclicLR(base_learning_rate=2.0e-3, max_learning_rate=5.0e-5, step_size_up=10)
    #lr = paddle.optimizer.lr.InverseTimeDecay(learning_rate=5.0e-3, gamma=0.25, last_epoch=-1)
    #lr = paddle.optimizer.lr.NoamDecay(d_model=n_hidden, warmup_steps=opt_steps)

    opt = paddle.optimizer.AdamW(learning_rate=lr, parameters=model.parameters(),
            grad_clip=paddle.nn.ClipGradByGlobalNorm(1.0),
            )
    opt = fleet.distributed_optimizer(opt)

    #load model
    if(load_model_path is not None):
        load_model(model, opt, load_model_path)


    #Add inner loss weights
    if(train_warmup_steps is not None):
        inner_loss_weights = linear_segments(L + 1, warmup_steps=train_warmup_steps, intermediate_steps=train_intermediate_steps)
    else:
        inner_loss_weights = None

    print("Loss weights:", inner_loss_weights)
    epoch = 0


    while epoch < MAX_EPOCH:
        batch_idx = 0
        epoch += 1
        epoch_loss = None
        epoch_toks = None
        # Training
        is_end = False
        train_stat = EpochStat()
        if(isinstance(DataGen, FileDataset)):
            DataGen.reset(batch_size)
        while batch_idx < MAX_ITERATION and not is_end:
            model.train()
            batch_idx += 1
            batch_loss, batch_toks, pos_loss, is_end = train_batch(
                    DataGen, 
                    model, 
                    pad_id,
                    worker_index,
                    batch_size=batch_size,
                    opt=opt,
                    loss_weights=inner_loss_weights,
                    seg_size=train_segment,
                    detach_segments=detach_segments,
                    is_train=True
                    )
            train_stat.add_batch(batch_toks, batch_loss, pos_loss, None)
            print("Epoch: %d, batch %d, learning rate: %f, average loss: %s"%(epoch, batch_idx, opt.get_lr(), formulate_numpyarray(batch_loss / batch_toks)))
            sys.stdout.flush()
        lr.step()
        epoch_loss, pos_loss, _ = train_stat.get_statistics()
        print("[TRAINING] Epoch: %d, training loss: %s"%(epoch, formulate_numpyarray(epoch_loss)))
        print("[TRAINING]" + "\t".join(map(lambda x:"%.3f"%x, pos_loss.tolist())))
        sys.stdout.flush()

        if(worker_index == 0):
            save_model(model, opt, "checkpoint.%s/epoch"%model_type, epoch)
        
        # Testing
        if(worker_index == 0):
            eval_loader.reset(batch_size)
            is_end = False
            epoch_loss = None
            epoch_toks = None
            model.eval()
            eval_stat = EpochStat()
            while not is_end:
                batch_loss, batch_toks, pos_loss, is_end = train_batch(
                        eval_loader, 
                        model, 
                        pad_id,
                        worker_index,
                        batch_size=batch_size,
                        seg_size=eval_segment,
                        is_train=False
                        )
                #print(batch_loss, batch_toks)
                eval_stat.add_batch(batch_toks, batch_loss, pos_loss, None)
            epoch_loss, pos_loss, _ = eval_stat.get_statistics()
            print("[EVALUATING] Epoch: %d, evaluation ppl: %s"%(epoch, formulate_numpyarray(epoch_loss)))
            print("[EVALUATING]" + "\t".join(map(lambda x:"%.3f"%x, pos_loss.tolist())))
            sys.stdout.flush()

