import hydra 
from omegaconf import DictConfig, OmegaConf
import pyrootutils
import matplotlib.pyplot as plt
from src import utils
import os
from pytorch_lightning import seed_everything
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
log = utils.get_pylogger(__name__)



#OmegaConf.register_new_resolver("set_env", set_env)

@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    #OmegaConf.resolve(cfg)
    log.info("Logging gpu info")
    utils.log_gpu_memory_metadata()

    if cfg.get("seed"):
        log.info("Setting seed")
        seed_everything(cfg.seed)
    log.info(f"Instantiating datamodule <<{cfg.datamodule._target_}>>")

    datamodule= hydra.utils.instantiate(cfg.datamodule,_recursive_=False)
    log.info(f"Instantiating Lit - model <<{cfg.backbone._target_}>>")

    model = hydra.utils.instantiate(cfg.backbone)
    log.info(f"Instantiating Trainer <<{cfg.trainer._target_}>>")
    datamodule.setup()
    trainer = hydra.utils.instantiate(cfg.trainer)

    log.info("Starting training")
    print(model)
    print(trainer)
    trainer.fit(model=model,datamodule=datamodule)

if __name__ == "__main__":
    main()