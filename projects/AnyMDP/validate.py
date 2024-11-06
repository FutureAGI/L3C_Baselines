import os
import sys
from l3c_baselines.models import AnyMDPRSA
from l3c_baselines.utils import Runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from anymdp_epoch import AnyMDPEpoch

if __name__ == "__main__":
    runner=Runner()
    runner.start(AnyMDPRSA, [], AnyMDPEpoch, extra_info='validate')