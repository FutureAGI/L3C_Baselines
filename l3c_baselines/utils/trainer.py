import os
import sys
import argparse
import torch
import numpy
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import wraps
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.amp import autocast, GradScaler
from l3c_baselines.dataloader.prefetch_dataloader import PrefetchDataLoader
from .tools import Configure, Logger, log_progress, log_debug, log_warn, log_fatal
from .tools import count_parameters, check_model_validity, model_path, safety_check, apply_gradient_safely, custom_load_model
from .scheduler import noam_scheduler

def EpochManager(cls):
    @wraps(cls, updated=())
    class WrapperEpochManager(object):
        def __init__(self, **kwargs):
            self.computer = cls(**kwargs)
            for key in kwargs:
                setattr(self, key, kwargs[key])

            
        def get(self, attr, config=None):
            if(hasattr(self.computer, attr)):
                return getattr(self.computer, attr)
            elif(config is not None):
                if(config.has_attr(attr)):
                    return getattr(self.config, attr)
                else:
                    return None
            else:
                return None

        def init_dataset(self):
            self.dataset = self.get('dataset')
            if(self.dataset is None):
                DataType = self.get('DataType')
                if(DataType is None):
                    log_fatal("Must define `DataType` to initialize the dataset")
                data = DataType(self.config.data_path, 
                                    self.config.seq_len, 
                                    verbose=self.main)
                self.dataset = PrefetchDataLoader(data, batch_size=self.config.batch_size, 
                                            rank=self.rank, world_size=self.world_size)
                self.computer.dataset = self.dataset

        def init_logger(self):
            self.logger_keys=None
            if(hasattr(self.computer, 'logger_keys') and len(self.computer.logger_keys)!=0):
                assert type(self.computer.logger_keys) == list, \
                    f"The logger_keys must be a list of string."
                self.logger_keys=self.computer.logger_keys

                if(self.is_training):
                    log_file = self.log_config.training_log
                    process_name = "Training"
                    max_iter = len(self.dataset)
                else:
                    log_file = self.log_config.evaluation_log
                    process_name = "Evaluation"
                    max_iter = -1

                self.logger = Logger(
                        *self.logger_keys,
                        on=self.main, 
                        max_iter=max_iter,
                        use_tensorboard=self.log_config.use_tensorboard,
                        log_file=log_file,
                        prefix=f"{self.run_name}-{process_name}",
                        field=f"{self.log_config.tensorboard_log}/{self.run_name}-{process_name}")
            else:
                self.logger=None
            self.computer.logger = self.logger

        def init_optimizer(self):
            if(self.is_training):
                self.optimizer = self.get('optimizer')
                if(self.optimizer is None):
                    lr = self.get('lr', config=self.config)
                    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
                    self.computer.optimizer = self.optimizer

                # Initialize the learning rate schedulers
                self.scheduler = self.get('lr_scheduler')
                if(self.scheduler is None):
                    lr_decay_interval = self.get('lr_decay_interval', config=self.config)
                    self.scheduler = LambdaLR(self.optimizer, 
                        lr_lambda=lambda x:noam_scheduler(x, lr_decay_interval))
                    self.computer.lr_scheduler = self.scheduler
                
                step = self.get('lr_start_step', config=self.config)
                if(step is None):
                    step = 0
                self.scheduler.step(0)

                self.scaler=None
                if(self.config.use_scaler):
                    self.scaler = GradScaler()
                self.computer.scaler = self.scaler

        def _epoch_start(self, eid):
            if(hasattr(self.computer, 'epoch_start')):
                self.computer.epoch_start(eid)
        
        def _epoch_end(self, eid):
            if(hasattr(self.computer, 'epoch_end')):
                self.computer.epoch_end(eid)

        def _preprocess(self):
            self.init_dataset()
            self.init_logger()
            self.init_optimizer()
            if(hasattr(self.computer, 'preprocess')):
                self.computer.preprocess()

        def _postprocess(self):
            if(hasattr(self.computer, 'postprocess')):
                self.computer.postprocess()

        def run(self, epoch_id, device, device_type):
            acc_iter = 0
            self.dataset.dataset.reset(epoch_id)

            if(not hasattr(self.computer, 'compute')):
                log_fatal("The computer object must have compute method.")

            data_length = len(self.dataset)
            for batch_id, batch_data in enumerate(self.dataset):
                acc_iter += 1

                # Important: Must reset the model before segment iteration
                self.model.module.reset()
                if(self.is_training):
                    self.optimizer.zero_grad()
                    with autocast(dtype=torch.bfloat16, enabled=self.config.use_amp, device_type=device_type):
                        self.computer.compute(device,
                                  *batch_data, 
                                  epoch_id=epoch_id, 
                                  batch_id=batch_id)
                    apply_gradient_safely(self.model, self.optimizer, scaler=self.scaler)
                    self.scheduler.step()
                else:
                    with torch.no_grad():
                        self.computer.compute(device,
                                  *batch_data, 
                                  epoch_id=epoch_id, 
                                  batch_id=batch_id)

                # Safety Check and Save
                need_break = False
                if(self.is_training and self.config.has_attr("max_save_iterations") 
                                and acc_iter > self.config.max_save_iterations 
                                and self.config.max_save_iterations > 0):
                    acc_iter = 0
                    log_debug("-"*40, "Check current validity and save model for safe...", on=self.main)
                    if(self.main):
                        check_model_validity(self.model.module)
                        save_model_path = model_path(self.config.save_model_path, epoch_id)
                        torch.save(self.model.state_dict(), save_model_path)
                    need_break = True

                if(not self.is_training):
                    log_progress((batch_id + 1) / data_length, on=self.main)

                yield need_break
            yield True
    return WrapperEpochManager

def dist_process(rank, use_gpu, world_size, config, main_rank,
                model_type, train_objects, evaluate_objects):
    """
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
        print("Main gpu", use_gpu, "rank:", rank, device)

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

    if(not isinstance(train_objects, list) and not isinstance(train_objects, tuple)):
        train_objects = [train_objects]
    if(not isinstance(evaluate_objects, list) and not isinstance(evaluate_objects, tuple)):
        evaluate_objects = [evaluate_objects]


    train_list = []
    for train_object in train_objects:
        train_list.append(train_object(run_name=config.run_name, 
                                        model=model, 
                                        config=config.train_config,
                                        log_config=config.log_config,
                                        rank=rank,
                                        world_size=world_size,
                                        device_type=device_type,
                                        device=device,
                                        main=main,
                                        is_training=True))

    evaluate_list = []
    for evaluate_object in evaluate_objects:
        evaluate_list.append(evaluate_object(run_name=config.run_name, 
                                        model=model, 
                                        config=config.test_config,
                                        log_config=config.log_config,
                                        rank=rank,
                                        world_size=world_size,
                                        device_type=device_type,
                                        device=device,
                                        main=main,
                                        is_training=False))

    for train_object in train_list:
        train_object._preprocess()
    for evaluate_object in evaluate_list:
        evaluate_object._preprocess()

    def evaluate_epoch(eid):
        for evaluate_object in evaluate_list:
            evaluate_object._epoch_start(eid)
            for _ in evaluate_object.run(eid, device, device_type):
                pass
            evaluate_object._epoch_end(eid)

    if(len(train_list) < 1):
        evaluate_epoch(0) # Doing single epoch evaluation
    else:
        epoch = 0
        while epoch < config.train_config.max_epochs:
            epoch += 1
            for train_object in train_list:
                train_object._epoch_start(epoch)
                for need_evaluate in train_object.run(epoch, device, device_type):
                    if(need_evaluate):
                        evaluate_epoch(epoch)
                train_object._epoch_end(epoch)

    for train_object in train_list:
        train_object._postprocess()
    for evaluate_object in evaluate_list:
        evaluate_object._postprocess()

class Runner(object):
    """
    Trainer class manage the training process and framework
    """
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('configuration', type=str, help="YAML configuration file")
        parser.add_argument('--configs', nargs='*', help="List of all configurations, overwrite configuration file: eg. train_config.batch_size=16 test_config.xxx=...")
        args = parser.parse_args()

        self.use_gpu = torch.cuda.is_available()
        self.world_size = torch.cuda.device_count() if self.use_gpu else os.cpu_count()
        if(self.use_gpu):
            print("Use Parallel GPUs: %s" % self.world_size)
        else:
            print("Use Parallel CPUs: %s" % self.world_size)

        self.config = Configure()
        self.config.from_yaml(args.configuration)

        # Get the dictionary of attributes
        if args.configs:
            for pair in args.configs:
                key, value = pair.split('=')
                self.config.set_value(key, value)
                print(f"Rewriting configurations from args: {key} to {value}")
        
        print("Final configuration:\n", self.config)

        if(self.config.has_attr('master_addr')):
            os.environ['MASTER_ADDR'] = self.config.master_addr
        else:
            os.environ['MASTER_ADDR'] = 'localhost' 

        os.environ['MASTER_PORT'] = self.config.master_port

    def start(self, model_type, train_objects, evaluate_objects):
        mp.spawn(dist_process,
                args=(self.use_gpu, 
                      self.world_size, 
                      self.config, 
                      0, # always use #0 as the main GPU
                      model_type,
                      train_objects, 
                      evaluate_objects),
                nprocs=self.world_size if self.use_gpu else min(self.world_size, 4),  # Limit CPU processes if desired
                join=True)