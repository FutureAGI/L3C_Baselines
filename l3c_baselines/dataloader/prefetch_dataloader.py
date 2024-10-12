import os
import sys
import random
import torch
import itertools
import queue
import numpy as np
import multiprocessing
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate as torch_collate
from restools.logging import log_warn, log_fatal

class NaiveDataLoader(DataLoader):
    def __init__(self, dataset, rank=0, world_size=1, batch_size=4, collate_fn=torch_collate):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.rank = rank
        self.world_size = world_size
        self.index = 0
        self.local_index = 0
        self.length = (len(self.dataset) - 1) // (self.batch_size * self.world_size) + 1
        self.data_volume = len(self.dataset)

    def __iter__(self):
        self.index = self.rank
        self.local_index = 0
        return self

    def __next__(self):
        if self.local_index >= self.length:
            # stop iteration once index is out of bounds
            raise StopIteration
        data = []
        sub_iter = 0
        while sub_iter < self.batch_size:
            # In case the data is invalid, fetch further data
            sub_data = self.get()
            if(sub_data is not None):
                data.append(sub_data)
                sub_iter += 1
        self.local_index += 1
        return self.collate_fn(data)

    def get(self):
        item = self.dataset[self.index]
        self.index += self.world_size
        return item

    def __len__(self):
        return self.length

def worker_fn(dataset, length, index_queue, output_queue):
    while True:
        try:
            index = index_queue.get(timeout=0)
        except queue.Empty:
            continue
        if index is None:
            break

        if index > length:
            # Allowing fetch index to exceed max length to accommodate certain mistake
            real_idx = index % length
        else:
            real_idx = index
        try:
            output_queue.put((index, dataset[real_idx]))
        except Exception:
            print(f"Unexpected error when getting index={index}, actual_index={real_idx}, put None")
            sys.stdout.flush()
            output_queue.put((index, None))

class PrefetchDataLoader(NaiveDataLoader):
    def __init__(
        self,
        dataset,
        rank=0,
        world_size=1,
        batch_size=4,
        num_workers=2,
        prefetch_batches=2,
        collate_fn=torch_collate,
    ):
        super().__init__(dataset, 
            batch_size=batch_size, 
            rank=rank, 
            world_size=world_size, 
            collate_fn=collate_fn)

        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.output_queue = multiprocessing.Queue()
        self.index_queues = []
        self.workers = []
        self.worker_cycle = itertools.cycle(range(num_workers))
        self.cache = {}
        self.prefetch_local_index = 0

        for _ in range(num_workers):
            index_queue = multiprocessing.Queue()
            worker = multiprocessing.Process(
                target=worker_fn, args=(self.dataset, self.length, index_queue, self.output_queue)
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            self.index_queues.append(index_queue)

    def prefetch(self):
        while (self.prefetch_index < self.index + self.prefetch_batches * self.batch_size * self.world_size):
            self.index_queues[next(self.worker_cycle)].put(self.prefetch_index)
            self.prefetch_index += self.world_size

    def get(self):
        self.prefetch()
        sys.stdout.flush()
        if self.index in self.cache:
            item = self.cache[self.index]
            del self.cache[self.index]
        else:
            while True:
                try:
                    (index, data) = self.output_queue.get(timeout=0)
                except queue.Empty:  # output queue empty, keep trying
                    continue
                if index == self.index:  # found our item, ready to return
                    item = data
                    break
                else:  # item isn't the one we want, cache for later
                    self.cache[index] = data
        sys.stdout.flush()

        self.index += self.world_size
        return item

    def __iter__(self):
        self.local_index = 0
        self.index = self.rank
        self.prefetch_index = self.index
        self.cache = {}
        self.prefetch()
        return self

    def __del__(self):
        try:
            # 通知每个工作进程停止工作
            for i in range(len(self.workers)):
                self.index_queues[i].put(None)
            
            # 等待所有工作进程结束
            for w in self.workers:
                w.join()
            
            # 关闭所有队列
            for q in self.index_queues:
                q.close()
                q.cancel_join_thread()
            self.output_queue.close()
            self.output_queue.cancel_join_thread()
        except Exception as e:
            # 在异常情况下，尝试终止所有工作进程
            for w in self.workers:
                if w.is_alive():
                    w.terminate()
            # 重新抛出异常以便上层处理
            raise e
    
def segment_iterator(full_len, seg_len, device, *args):
    """
    Input shape: [Batch, Length, *]
    Output: list of [Batch, seg_len, *]
    """
    arg_ext = []

    # Make sure full_len is shorter than args
    for arg in args:
        arg_len = arg.shape[1]
        if(full_len > arg_len):
            full_len = min(full_len, arg_len)

    for arg in args:
        arg_len = arg.shape[1]
        if(arg_len == full_len + 1):
            arg_ext.append(True)
        elif(arg_len == full_len):
            arg_ext.append(False)
        else:
            log_fatal(f"Dataloader - segement_iterator: revised length {full_len} is not matched with {arg_len}")

    seg_num = (full_len - 1) // seg_len + 1
    for seg_id in range(seg_num):
        res = [seg_id]
        b = seg_id * seg_len
        e = min(b + seg_len, full_len)
        for i,arg in enumerate(args):
            if(arg_ext[i]):
                res.append(arg[:, b:e+1].to(device))
            else:
                res.append(arg[:, b:e].to(device))
        yield tuple(res)
