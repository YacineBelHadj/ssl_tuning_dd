   #%% 
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
from src.utils import task_wrapper

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

@task_wrapper
def train(cfg: DictConfig):
    #OmegaConf.resolve(cfg)

    log.info("Logging gpu info")
    utils.log_gpu_memory_metadata()

    if cfg.get("seed"):
        log.info("Setting seed")
        seed_everything(cfg.seed)
    
    log.info(f"Instantiating datamodule <<{cfg.datamodule._target_}>>")
    datamodule= hydra.utils.instantiate(cfg.datamodule,_recursive_=False)
    datamodule.setup()
    train = datamodule.train_dataloader()
    print(len(train.dataset[0]))
    log.info(f'Instantiating callbacks <<......>>')
    callbacks = inst.instantiate_callbacks(cfg.callbacks)

    log.info(f'Instantiating loggers <<......>>')
    logger = inst.instantiate_loggers(cfg.logger)    

    log.info(f"Instantiating Lit - model <<{cfg.backbone._target_}>>")
    modelmodule = hydra.utils.instantiate(cfg.backbone)

    log.info(f"Instantiating Trainer <<{cfg.trainer._target_}>>")
    trainer = hydra.utils.instantiate(cfg.trainer,callbacks=callbacks,logger=logger)

    log.info("Starting training")

    trainer.fit(model=modelmodule,datamodule=datamodule)
    # save file with hi at hi.txt
    os.system("echo hi > hi.txt")
    log.info("Training complete")
    log.info("Starting testing")
    trainer.test(model=modelmodule,datamodule=datamodule)
    
    log.info(f"Instantiating Downstream task <<{cfg.downstream._target_}>>")
    dds = hydra.utils.instantiate(cfg.downstream,encoder=modelmodule.model.encoder)
    dds.fit(datamodule.train_dataloader())

    log.info(f"Instantiating benchmark <<{cfg.eval._target_}>>")
    #%% 
    benchmark = hydra.utils.instantiate(cfg.eval,dds=dds,datamodule=datamodule)
    log.info("Starting benchmark")
    benchmark.setup()
    res  = benchmark.evaluate()
    return res

@hydra.main(**_HYDRA_PARAMS)
def main(cfg: DictConfig):
    log.info("Starting training script")
    log.info("Setting environment variables")
    setting_environment(cfg.env)
    train(cfg)

    
if __name__ == "__main__":
    main()