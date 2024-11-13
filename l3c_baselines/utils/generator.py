import os
import sys
import argparse
import torch
import numpy
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from .tools import  Logger, log_progress, log_debug, log_warn, log_fatal
from .tools import custom_load_model
from .trainer import Runner


class GeneratorBase(object):
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])
        self.T_ini = self.config.decoding_strategy.T_ini
        self.T_fin = self.config.decoding_strategy.T_fin
        self.T_decay_type = self.config.decoding_strategy.decay_type
        self.max_steps = self.config.max_steps

        self.dT_linear = (self.T_fin - self.T_ini) / self.max_steps
        self.dT_exp = numpy.exp((numpy.log(self.T_fin) - numpy.log(self.T_ini)) / self.max_steps)

    def _scheduler(self, step, type=None):
        # A inner built scheduler for decoding strategy
        if(type is None):
            type = self.T_decay_type
        if(type.lower() == "linear"):
            return self.T_ini + step * self.dT_linear
        elif(type.lower() == "exponential"):
            return self.T_ini * (self.dT_exp ** step)
        else:
            log_fatal(f"Unknown decoder type {type}")

    def preprocess(self):
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("__call__ method must be implemented")

    def postprocess(self):
        pass

def dist_generator(rank, use_gpu, world_size, config, main_rank,
                model_type, generator_class, extra_info):
    """
    Distributed generator 
    """
    if use_gpu:
        torch.cuda.set_device(rank)  # Set the current GPU to be used
        device = torch.device(f'cuda:{rank}')
        device_type = 'cuda'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if(main_rank is None):
        main = False
    elif(main_rank == "all" or main_rank == rank):
        main = True
    else:
        main = False

    if(main):
        log_debug("Main gpu", use_gpu, "rank:", rank, device)

    # Create model and move it to GPU with id `gpu`
    model = model_type(config.model_config, verbose=main)
    model = model.to(device)

    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    # Load the model if specified in the configuration
    if(config.has_attr("load_model_path") and 
            config.load_model_path is not None and 
            config.load_model_path.lower() != 'none'):
        if(config.has_attr("load_model_parameter_blacklist")):
            black_list = config.load_model_parameter_blacklist
        else:
            black_list = []
        model = custom_load_model(model, f'{config.load_model_path}/model.pth', 
                                  black_list=black_list,
                                  verbose=main, 
                                  strict_check=False)
    else:
        log_warn("No model is loaded as `load_model_path` is not found in config or is None", on=main)

    generator=generator_class(run_name=config.run_name, 
                            model=model, 
                            config=config.generator_config,
                            rank=rank,
                            world_size=world_size,
                            device_type=device_type,
                            device=device,
                            main=main,
                            extra_info=extra_info)
    generator.preprocess()
    for key in config.generator_config.epoch_numbers:
        generator()
    generator.postprocess()

class GeneratorRunner(Runner):
    """
    Generator class manage the interaction process and framework
    """
    def start(self, model_type, generator_class, extra_info=None):
        mp.spawn(dist_generator,
                args=(self.use_gpu, 
                      self.world_size, 
                      self.config, 
                      0, # always use #0 as the main GPU
                      model_type,
                      generator_class,
                      extra_info),
                nprocs=self.world_size if self.use_gpu else min(self.world_size, 4),  # Limit CPU processes if desired
                join=True)