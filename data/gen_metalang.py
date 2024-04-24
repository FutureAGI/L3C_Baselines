#   Copyright (c) 2021 DeepEvolution Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is used to generate data for meta language models

import sys
import argparse
import multiprocessing
import numpy
import random
from l3c.metalang import MetaLangV1
from l3c.metalang import MetaLangV2

def dump_data(path, idxes, args):
    vocab_size = random.choice(list(map(int, args.vocab_size.split(","))))
    n_gram = random.choice(list(map(int, args.n_gram.split(","))))
    embedding_size = random.choice(list(map(int, args.embedding_size.split(","))))
    hidden_size = random.choice(list(map(int, args.hidden_size.split(","))))
    dataset = MetaLangV2(
            V=vocab_size,
            n=n_gram,
            d=embedding_size,
            N=hidden_size,
            e=args.error_rate,
            L=args.sequence_length)
    for idx in idxes:
        tokens = dataset.generate_npy(args.file_size, "%s/lm_%05d.npy"%(path, idx))
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generating Pseudo-Training Data')
    parser.add_argument('--vocab_size', type=str, default="64, 128")
    parser.add_argument('--embedding_size', type=str, default="16, 32")
    parser.add_argument('--hidden_size', type=str, default="16, 32, 64")
    parser.add_argument('--elements_length', type=int, default=64)
    parser.add_argument('--elements_number', type=int, default=10)
    parser.add_argument('--error_rate', type=float, default=0.20)
    parser.add_argument('--n_gram', type=str, default="2,3,4,5")
    parser.add_argument('--sequence_length', type=int, default=4096)
    parser.add_argument('--file_size', type=int, default=1000)
    parser.add_argument('--file_number', type=int, default=1024)
    parser.add_argument("--workers", type=int, default=4, help="number of multiprocessing workers")
    parser.add_argument('--output_path', type=str, default="./metalm_data")

    args = parser.parse_args()


    processes = []
    n_b_t = 0
    worker_splits = args.file_number // args.workers
    for worker_id in range(args.workers):
        n_e_t = n_b_t + worker_splits
        n_b = int(n_b_t)
        n_e = int(n_e_t)

        print("start processes generating %05d to %05d" % (n_b, n_e))
        process = multiprocessing.Process(target=dump_data, 
                args=(args.output_path, range(n_b, n_e), args))
        processes.append(process)
        process.start()

        n_b_t = n_e_t

    for process in processes:
        process.join() 
