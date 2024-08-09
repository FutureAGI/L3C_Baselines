"""
Logger Tools
"""

import sys
import time
import numpy
import torch
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, *args, sum_iter=-1, use_tensorboard=False):
        self.keys = []
        self.sum_iter = sum_iter
        if(use_tensorboard):
            self.writer = SummaryWriter()
        else:
            self.writer = None
        for arg in args:
            self.keys.append(arg)

    def log(self, *args, epoch=-1, iteration=-1, prefix=None, rewrite=False):
        if(len(args) != len(self.keys)):
            raise Exception(f"Mismatch logger key (n={len(self.keys)}) and value (n={len(args)})")

        log = time.strftime("%Y%m%d-%H:%M:%S", time.localtime())
        if(prefix is not None):
            log +=f'[{prefix}]'
        if(epoch > -1):
            log += ' ' + "Epoch:%03d"%epoch
        if(iteration > -1 and self.sum_iter > -1):
            progress = (iteration + 1) / self.sum_iter * 100
            log += ' ' + "Progress:%.3f%%"%progress

        def format_value(x):
            if(isinstance(x, float)):
                if((abs(x) < 0.10 or abs(x) > 1.0e+3)):
                    return "%.3e"%x
                else:
                    return "%.3f"%x
            elif(isinstance(x, int)):
                return "%03d"%x
            elif(isinstance(x, list) or isinstance(x, numpy.ndarray)):
                return ",".join(map(format_value, x))
            else:
                return str(x)
            

        for i, arg in enumerate(args):
            log += ' ' + self.keys[i] + ":" + format_value(arg)
            if(self.writer is not None and iteration > -1):
                self.writer.add_scalar(self.keys[i], float(arg), iteration)
        if(rewrite):
            log += '\r'
        else:
            log += '\n'
        sys.stdout.write(log)
        sys.stdout.flush()


# Paint a progress bar
def show_bar(fraction, bar):
    percentage = int(bar * fraction)
    empty = bar - percentage
    sys.stdout.write("[" + "=" * percentage + " " * empty + "]" + "%.2f %%\r" % (percentage * 100 / bar))
    sys.stdout.flush()
