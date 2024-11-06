import os
import sys
from l3c_baselines.models import E2EObjNavSA
from l3c_baselines.utils import Runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from maze_epoch import MazeEpochVAE, MazeEpochCausal

if __name__ == "__main__":
    runner=Runner()
    runner.start(E2EObjNavSA, [], [MazeEpochVAE, MazeEpochCausal]])