import os
import sys
from l3c_baselines.models import LanguageModel
from l3c_baselines.utils import Runner
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from lm_epoch import LMEpoch

if __name__ == "__main__":
    runner=Runner()
    runner.start(LanguageModel, LMEpoch, LMEpoch, extra_info='validate')