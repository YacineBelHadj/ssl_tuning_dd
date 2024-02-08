import torch
from src import utils
import random
import numpy as np
import os
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlInit,
)

log = utils.get_pylogger(__name__)


def log_gpu_memory_metadata() -> None:
    """Logging GPUs memory metadata (total, free and used) if it's available by
    PYNVML."""
    gpus_num = torch.cuda.device_count()
    if gpus_num == 0:
        return
    nvmlInit()
    cards = (nvmlDeviceGetHandleByIndex(num) for num in range(gpus_num))
    for i, card in enumerate(cards):
        info = nvmlDeviceGetMemoryInfo(card)
        log.info(f"GPU memory info: card {i} : total : {info.total}")
        log.info(f"GPU memory info: card {i} : free  : {info.free}")
        log.info(f"GPU memory info: card {i} : used  : {info.used}")



def set_seed(
    seed: int = 42, deterministic: bool = True, benchmark: bool = False
) -> None:
    """Manually set seeds, deterministic and benchmark modes.

    Included seeds:
        - random.seed
        - np.random.seed
        - torch.random.manual_seed
        - torch.cuda.manual_seed
        - torch.cuda.manual_seed_all

    Also, manually set up deterministic and benchmark modes.

    Args:
        seed (int): Seed. Default to 42.
        deterministic (bool): deterministic mode. Default to True.
        benchmark (bool): benchmark mode. Default to False.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.cuda.deterministic = deterministic
        torch.cuda.benchmark = benchmark

def setting_environment(cfg):
    """Set environment variables from config."""
    assert len(cfg) < 10, "Too many environment variables to set! are you sure you want to set all of them?"

    for key, value in cfg.items():
        # check string 
        if not isinstance(value, str):
            raise TypeError(f"Environment variable <{key}> must be a string!")
        
        os.environ[key] = value
        log.info(f"Set environment variable: {key}={value}")
