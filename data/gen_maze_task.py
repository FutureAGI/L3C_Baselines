import argparse
import pickle
import random
import l3c.mazeworld
from l3c.mazeworld import MazeTaskSampler
from l3c.mazeworld.agents import SmartSLAMAgent
from l3c.mazeworld.envs.maze_task import MazeTaskManager


if __name__=="__main__":
    # Parse the arguments, should include the output file name
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=str, default="9,15,21,25,31,35", help="a list of scales separated with comma to randomly choose")
    parser.add_argument("--density", type=str, default="0.20,0.34,0.36,0.38,0.45", help="density:a list of float")
    parser.add_argument("--landmarks", type=str, default="5,6,7,8,9,10", help="landmarks:a list of number of landmarks")
    parser.add_argument("--task_number", type=int, default=1, help="multiple epochs:default:1")
    parser.add_argument("--output_path", type=str, help="output file name")
    args = parser.parse_args()

    #used for dump tasks only
    tasks = []
    scales = list(map(int, args.scale.split(',')))
    densities = list(map(float, args.density.split(',')))
    landmarks = list(map(int, args.landmarks.split(',')))
    for idx in range(args.task_number):
        scale = random.choice(scales)
        density = random.choice(densities)
        n_landmark = random.choice(landmarks)
        print(f"Generating tasks with n={scale}, density={density}, count_landmarks={n_landmark} ...")
        task = MazeTaskSampler(n=scale, allow_loops=True, 
                wall_density=density,
                landmarks_number=n_landmark,
                landmarks_avg_reward=0.5,
                commands_sequence = 10000,
                verbose=False)
        task_dict = task._asdict()
        tasks.append(task_dict)
    if(args.output_path.find('.pkl') < 0):
        output_file = args.output_path + ".pkl"
    else:
        output_file = args.output_path
    print(f"Writing tasks to {output_file} with following configuration")
    with open(output_file, 'wb') as fw:
        pickle.dump(tasks, fw)
