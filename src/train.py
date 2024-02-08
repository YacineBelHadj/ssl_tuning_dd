import comet
import hydra 
from omegaconf import DictConfig, OmegaConf
import pyrootutils
from src import utils
import os
from pytorch_lightning import seed_everything
from src.utils.resolve import set_env
from src.utils import instantiator as inst
from src.utils import setting_environment

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
    #OmegaConf.resolve(cfg)
    log.info("Starting training script")
    log.info("Setting environment variables")
    print(cfg.env)
    setting_environment(cfg.env)
    log.info("Logging gpu info")
    utils.log_gpu_memory_metadata()

    if cfg.get("seed"):
        log.info("Setting seed")
        seed_everything(cfg.seed)
    
    log.info(f"Instantiating datamodule <<{cfg.datamodule._target_}>>")
    datamodule= hydra.utils.instantiate(cfg.datamodule,_recursive_=False)

    log.info(f'Instantiating callbacks <<......>>')
    callbacks = inst.instantiate_callbacks(cfg.callbacks)

    log.info(f'Instantiating loggers <<......>>')
    logger = inst.instantiate_loggers(cfg.logger)    

    log.info(f"Instantiating Lit - model <<{cfg.backbone._target_}>>")
    model = hydra.utils.instantiate(cfg.backbone)

    log.info(f"Instantiating Trainer <<{cfg.trainer._target_}>>")
    trainer = hydra.utils.instantiate(cfg.trainer,callbacks=callbacks,logger=logger)

    log.info("Starting training")

    trainer.fit(model=model,datamodule=datamodule)
    # save file with hi at hi.txt
    os.system("echo hi > hi.txt")
    log.info("Training complete")
    log.info("Starting testing")
    trainer.test(model=model,datamodule=datamodule)
    print(model.model.encoder)
    
if __name__ == "__main__":
    main()