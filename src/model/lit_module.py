from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from omegaconf import OmegaConf

class BaseLitModule(LightningModule):
    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        loss: DictConfig,
        metric: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """BaseLightningModule.

        Args:
            network (DictConfig): Network config.
            optimizer (DictConfig): Optimizer config.
            scheduler (DictConfig): Scheduler config.
            logging (DictConfig): Logging config.
            args (Any): Additional arguments for pytorch_lightning.LightningModule.
            kwargs (Any): Additional keyword arguments for pytorch_lightning.LightningModule.
        """
        super().__init__(*args, **kwargs)
        self.model = hydra.utils.instantiate(network)
        self.opt_params = optimizer
        self.slr_params = scheduler
        self.loss = hydra.utils.instantiate(loss)
        self.metric = {stage: hydra.utils.instantiate(metric) for stage in ["train", "val", "test"]}

    def forward(self, x: Any) -> Any:
        return self.model.forward(x)

    def configure_optimizers(self) -> Any:
        optimizer= hydra.utils.instantiate(self.opt_params, params=self.parameters(),_recursive_=True)
        if self.slr_params.get("scheduler"):
            scheduler: torch.optim.lr_scheduler = hydra.utils.instantiate(
                self.slr_params.scheduler,
                optimizer=optimizer,
                _convert_="partial",
            )
            lr_scheduler_dict = {"scheduler": scheduler}
            if self.slr_params.get("extras"):
                for key, value in self.slr_params.get("extras").items():
                    lr_scheduler_dict[key] = value
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        return {"optimizer": optimizer}
