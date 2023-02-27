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
        src_mask=None,
        pe_out=None,
        mems=None):
    # Mask the loss for Labels
    output, new_mems = model.forward(features, src_mask=src_mask, pe_out=pe_out, mems=mems)
    loss = F.cross_entropy(output, labels, weight=loss_mask, reduction="none")

    return loss, new_mems

def autoregressive_mask(data):
    length = data.shape[1]
    return paddle.tile(paddle.tensor.tril(
            paddle.full(shape=[length, length],
            fill_value=1,
            dtype=paddle.get_default_dtype())),
            [data.shape[0], 1, 1]
            )

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

    def add_batch(self, batch_toks, batch_loss, batch_mems):
        if(self.epoch_loss is None):
            self.epoch_toks = batch_toks
            self.epoch_loss = batch_loss
            if(batch_mems is not None):
                self.epoch_mems = batch_mems
                self.cnt += 1
        else:
            assert self.epoch_toks.shape[0] == batch_toks.shape[0], "statistics must be equal"
            self.epoch_toks += batch_toks
            self.epoch_loss += batch_loss
            if(self.epoch_mems is not None and batch_mems is not None):
                self.epoch_mems[:batch_toks.shape[0]] += batch_mems
                self.cnt += 1

    def get_statistics(self):
        if(self.cnt > 0):
            return (self.epoch_loss / self.epoch_toks), (1.0 / self.cnt) * self.epoch_mems
        else:
            return (self.epoch_loss / self.epoch_toks), None

