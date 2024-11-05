from l3c_baselines.models import AnyMDPRSA
from .anymdp_epoch import AnyMDPEpochTrainValidate
from l3c_baselines.utils import Runner

if __name__ == "__main__":
    runner=Runner()
    runner.start(AnyMDPRSA, AnyMDPEpochTrainValidate, AnyMDPEpochTrainValidate)