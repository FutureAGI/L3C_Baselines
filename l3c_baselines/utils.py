import numpy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

def detached_memory(mems):
    if(mems is None):
        return None
    elif(isinstance(mems, paddle.Tensor)):
        return mems.detach()
    elif(isinstance(mems, tuple)):
        return tuple([detached_memory(mem) for mem in mems])
    elif(isinstance(mems, list)):
        return [detached_memory(mem) for mem in mems]
    else:
        raise Exception("Unexpected type of memory: %s"%type(mems))

def metalm_loss_func(model, features,
        labels,
        loss_mask=None,
        mems=None):
    # Mask the loss for Labels
    output, new_mems = model.forward(features, mems=mems)
    loss = F.cross_entropy(output, labels, weight=loss_mask, reduction="none")

    return loss, new_mems

def linear_segments(length,
        minimum=0.0, 
        warmup_steps=128, 
        maximum=1.0, 
        intermediate_steps=128):
    deta = (maximum - minimum) / max(intermediate_steps, 1)
    seq = [minimum] * warmup_steps + [(i + 1) * deta + minimum for i in range(intermediate_steps)]
    if(length < len(seq)):
        return paddle.to_tensor(seq[:length], dtype="float32")
    else:
        return paddle.to_tensor(seq + [maximum] * (length - len(seq)), dtype="float32")

def formulate_numpyarray(array):
    return "\t".join(map(lambda x:"%.3f"%x, array.tolist()))


class EpochStat(object):
    def __init__(self):
        self.epoch_loss = None
        self.epoch_toks = None
        self.epoch_mems = None
        self.cnt = 0
        self.add_cnt = 0

    def add_batch(self, batch_toks, batch_loss, pos_loss, batch_mems):
        if(self.epoch_loss is None):
            self.epoch_toks = batch_toks
            self.epoch_loss = batch_loss
            self.epoch_pos_loss = pos_loss
            self.add_cnt = 1
            if(batch_mems is not None):
                self.epoch_mems = batch_mems
                self.cnt += 1
        else:
            assert self.epoch_toks.shape[0] == batch_toks.shape[0], "statistics must be equal"
            self.epoch_toks += batch_toks
            self.epoch_loss += batch_loss
            self.epoch_pos_loss += pos_loss
            self.add_cnt += 1
            if(self.epoch_mems is not None and batch_mems is not None):
                self.epoch_mems[:batch_toks.shape[0]] += batch_mems
                self.cnt += 1

    def get_statistics(self):
        if(self.cnt > 0):
            mem_sta = (1.0 / self.cnt) * self.epoch_mems
        else:
            mem_sta = None
        return (self.epoch_loss / self.epoch_toks), (1.0 / self.add_cnt) * self.epoch_pos_loss, mem_sta

def debug_print_grad(model):
    for name, parameter in model.named_parameters():
        print("gradient", name, "shape", parameter.shape, 
            "norm-inf", float(paddle.linalg.norm(parameter.grad, p=numpy.inf)))

def debug_print_para(model):
    sum_para_n = 0
    for name, parameter in model.named_parameters():
        sum_para_n += numpy.product(parameter.shape)
        print("parameter", name, "shape", parameter.shape, 
            "norm-inf", float(paddle.linalg.norm(parameter, p=numpy.inf)))
    print("total parameters: ", sum_para_n)

def debug_print_norm(mems, name=None):
    if(isinstance(mems, list) or isinstance(mems, tuple)):
        print("---- A-list of tensors:", name)
        for mem in mems:
            debug_print_norm(mem, name=name)
        print("---- End")
    elif(isinstance(mems, paddle.Tensor)):
        if(name is None):
            name = mems.name
        print("Paddle.Tensor: ", name, 
            "\tShape:", mems.shape, 
            "\tnorm-inf", float(paddle.linalg.norm(mems, p=numpy.inf)), 
            "\tnorm-l2",  float(paddle.linalg.norm(mems, p=2) / numpy.sqrt(numpy.prod(mems.shape))))

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
