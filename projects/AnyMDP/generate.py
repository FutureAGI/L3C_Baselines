import os
import sys
from l3c_baselines.models import AnyMDPRSA
from l3c_baselines.utils import GeneratorRunner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from anymdp_epoch import AnyMDPGenerator, MultiAgentGenerator

if __name__ == "__main__":
    runner=GeneratorRunner()
    runner.start(AnyMDPRSA, AnyMDPGenerator)