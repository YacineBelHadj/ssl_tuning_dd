import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Any
import pyrootutils
from src.utils.resolve import set_env

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
_HYDRA_PARAMS = {
    "version_base": "1.3",
    "config_path": str(root / "configs"),
    "config_name": "train.yaml",
}
#OmegaConf.register_new_resolver("set_env", set_env)
import matplotlib.pyplot as plt
@hydra.main(**_HYDRA_PARAMS)
def main(cfg):
    transform = hydra.utils.instantiate(cfg.transform)
    dataset = hydra.utils.instantiate(cfg.dataset)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
if __name__ == "__main__":
    main()