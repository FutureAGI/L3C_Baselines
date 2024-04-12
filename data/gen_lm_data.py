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
from l3c.metalm import MetaLMv1
from l3c.metalm import MetaLMv2

def dump_data(path, idxes, args):
    if(args.version == 'v1'):
        dataset = MetaLMv1(
                V=args.vocab_size,
                n=args.elements_number,
                l=args.elements_length,
                e=args.error_rate,
                L=args.sequence_length)
    elif(args.version == 'v2'):
        dataset = MetaLMv2(
                V=args.vocab_size,
                n=args.n_gram,
                d=args.embedding_size,
                N=args.hidden_size,
                e=args.error_rate,
                L=args.sequence_length)
    for idx in idxes:
        batch_fea, batch_lab = dataset.batch_generator(args.file_size)
        numpy.save("%s/lm_%05d.npy"%(path, idx), numpy.array([batch_fea, batch_lab]))
    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generating Pseudo-Training Data')
    parser.add_argument('--version', type=str, default='v1')
    parser.add_argument('--vocab_size', type=int, default=64)
    parser.add_argument('--embedding_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--elements_length', type=int, default=64)
    parser.add_argument('--elements_number', type=int, default=10)
    parser.add_argument('--error_rate', type=float, default=0.20)
    parser.add_argument('--n_gram', type=float, default=3)
    parser.add_argument('--sequence_length', type=int, default=4096)
    parser.add_argument('--samples', type=int, default=100)
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
