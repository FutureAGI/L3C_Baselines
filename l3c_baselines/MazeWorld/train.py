import argparse
import os
import torch
import numpy
from reader import read_data
from models import Models
import torch.optim as optim

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(data_path, file_names, model, optimizer, batch_size, time_step):
    total_iteration = 0
    model.debug_hooks_enabled = True
    for file_name in file_names:
        joint_data_path = os.path.join(data_path, file_name)
        observations, actions, rewards, maps = read_data(joint_data_path)
        epoch_size = observations.shape[0]
        observations = observations.transpose(0, 3, 1, 2)
        maps = maps.transpose(0, 3, 1, 2)
        _, C, W, H = observations.shape
        _, nC, nW, nH = maps.shape
        n_b = 0
        n_e = batch_size * time_step
        add_idxes = numpy.asarray([time_step * i for i in range(batch_size)], dtype=numpy.int32)
        while n_e < epoch_size - 1: # Give up those last few data
            observations_batch = observations[n_b:n_e].reshape(batch_size, time_step, C, W, H)
            observations_add = observations[n_b + add_idxes].reshape(batch_size, 1, C, W, H)
            observations_batch = numpy.concatenate((observations_batch, observations_add), axis=1)
            observations_batch = torch.from_numpy(observations_batch.astype("float32"))
            actions_batch = torch.from_numpy(actions[n_b:n_e].astype("int64")).view(batch_size, time_step)
            rewards_batch = torch.from_numpy(rewards[n_b:n_e].astype("float32")).view(batch_size, time_step)
            maps_batch = torch.from_numpy(maps[n_b:n_e]).view(batch_size, time_step, nC, nW, nH)

            try:
                loss = model.train_loss(observations_batch, actions_batch, rewards_batch, maps_batch)
                optimizer.zero_grad()
                loss.backward()
                prt_loss = loss.detach().numpy()
                total_iteration += 1
                print("Iteration: %s; Current File: %s; Loss: %s" % (total_iteration, file_name, prt_loss))
                sys.stdout.flush()

                # Print gradients for debugging
                #for name, param in model.named_parameters():
                #    if param.grad is not None:
                #        print(name, param.grad.norm().item())  # Print gradient norm

                optimizer.step()
                n_b += batch_size * time_step
                n_e += batch_size * time_step
            except RuntimeError as e:
                print("RuntimeError during backward pass:", e)
                # Additional debugging steps or handling of the error
                break  # Exit the loop or handle the error appropriately

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--time_step', type=int, default=32)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    file_names = os.listdir(args.data_path)
    model = Models(image_size=256, map_size=7, action_size=5)
    print("Number of parameters: ", count_parameters(model))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_epoch(args.data_path, file_names, model, optimizer, args.batch_size, args.time_step)
