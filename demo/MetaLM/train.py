import os
import sys
import argparse
import torch
import numpy
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from l3c_baselines.dataloader import LMDataSet, PrefetchDataLoader, segment_iterator
from l3c_baselines.utils import custom_load_model, noam_scheduler, LinearScheduler
from l3c_baselines.utils import show_bar, model_path
from l3c_baselines.utils import Configure, Logger, gradient_failsafe, DistStatistics
from l3c_baselines.models import LanguageModel
from l3c_baselines.utils import Logger, log_progress, log_debug, log_warn, log_fatal

os.environ['MASTER_ADDR'] = 'localhost'  # Example IP address, replace with your master node's IP

def main_epoch(rank, use_gpu, world_size, config, main_rank):
    if use_gpu:
        torch.cuda.set_device(rank)  # Set the current GPU to be used
        device = torch.device(f'cuda:{rank}')
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        device = torch.device('cpu')
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
    model = LanguageModel(config.model_config, verbose=main)

    model = model.to(device)

    if use_gpu:
        model = DDP(model, device_ids=[rank])
    else:
        model = DDP(model)

    # Example dataset and dataloader
    train_config = config.train_config
    test_config = config.test_config

    dataset = LMDataSet(train_config.data_path, train_config.file_size, verbose=main)
    dataloader = PrefetchDataLoader(dataset, batch_size=train_config.batch_size, rank=rank, world_size=world_size)

    logger = Logger("learning_rate", "perplexity", 
                on=main, max_iter=len(dataloader), use_tensorboard=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda x:noam_scheduler(x, train_config.lr_decay_interval))

    if(train_config.has_attr("load_model_path") and 
            train_config.load_model_path is not None and 
            train_config.load_model_path.lower() != 'none'):
        model = custom_load_model(model, 
                                  f'{train_config.load_model_path}/model.pth', 
                                  strict_check=False)

    # Perform the first evaluation
    test_config = config.test_config
    test_epoch(rank, use_gpu, world_size, test_config, model, main, device, 0)
    scaler = GradScaler()
    scheduler.step(train_config.lr_start_step)

    def main_round(rid, dataloader):
        total_iteration = len(dataloader)
        acc_iter = 0

        for batch_idx, (feature, label) in enumerate(dataloader):
            acc_iter += 1
            # Important: Must reset the model before each segment
            model.module.reset()
            start_position = 0
            for sub_idx, fea, lab in segment_iterator(
                    train_config.seq_len, train_config.seg_len, device, feature, label):
                fea = fea.to(device)
                lab = lab.to(device)
                optimizer.zero_grad()
                with autocast(dtype=torch.bfloat16, enabled=train_config.use_amp):
                    loss = model.module.perplexity(fea, lab, start_position=start_position)
                start_position += fea.shape[1]

                if(train_config.use_scaler):
                    scaler.scale(loss).backward()
                    gradient_failsafe(model.module, optimizer, scaler)
                    clip_grad_norm_(model.module.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                scheduler.step()

                logger(scheduler.get_last_lr()[0], 
                       float(loss.detach().cpu().numpy()), 
                       iteration=batch_idx, 
                       epoch=epoch_id)

    # Example training loop
    for epoch_id in range(1, train_config.max_epochs + 1):
        model.train()
        main_round(epoch_id, dataloader)
        if(main and epoch_id % train_config.evaluation_interval == 0):
            mod_path, opt_path_vae, opt_path_seq = model_path(train_config.save_model_path, epoch_id)
            torch.save(model.state_dict(), mod_path)
            #torch.save(optimizer.state_dict(), opt_path_seq)

        model.eval()
        # Perform the evaluation according to interval
        if(epoch_id % train_config.evaluation_interval == 0):
            test_epoch(rank, use_gpu, world_size, test_config, model, main, device, epoch_id)

def test_epoch(rank, use_gpu, world_size, config, model, main, device, epoch_id):
    # Example training loop

    results = []
    dataset = LMDataSet(config.data_path, config.file_size, verbose=main)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
    all_length = len(dataloader)

    if(main):
        log_debug("Start Evaluation...")
        log_progress(0)
    stat = DistStatistics("perplexity", "count")

    for batch_idx, (feature, label) in enumerate(dataloader):
        model.module.reset()
        start_position = 0
        for sub_idx, fea, lab in segment_iterator(
                        config.seq_len, config.seg_len, device, feature, label):
            fea = fea.to(device)
            lab = lab.to(device)
            with torch.no_grad():
                loss = model.module.perplexity(fea, lab)
                cnt = torch.tensor(fea.shape[0] * fea.shape[1]).to(loss.device)

            stat.add_with_safety(rank, 
                                perplexity=loss, 
                                count=cnt)

        if(main):
            log_progress((batch_idx + 1) / all_length)

    if(main):
        stat_res = stat()
        logger = Logger(*stat_res.keys())
        logger(*stat_res.values(), epoch=epoch_id)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('configuration', type=str, help="YAML configuration file")
    parser.add_argument('--configs', nargs='*', help="List of all configurations, overwrite configuration file: eg. train_config.batch_size=16 test_config.xxx=...")
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if use_gpu else os.cpu_count()
    if(use_gpu):
        print("Use Parallel GPUs: %s" % world_size)
    else:
        print("Use Parallel CPUs: %s" % world_size)

    config = Configure()
    config.from_yaml(args.configuration)

    # Get the dictionary of attributes
    if args.configs:
        for pair in args.configs:
            key, value = pair.split('=')
            config.set_value(key, value)
            print(f"Rewriting configurations from args: {key} to {value}")
    print("Final configuration:\n", config)
    os.environ['MASTER_PORT'] = config.train_config.master_port        # Example port, choose an available port

    mp.spawn(main_epoch,
             args=(use_gpu, world_size, config, 0),
             nprocs=world_size if use_gpu else min(world_size, 4),  # Limit CPU processes if desired
             join=True)
