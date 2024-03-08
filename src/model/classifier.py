from src.model.lit_module import BaseLitModule
import hydra
from omegaconf import DictConfig
from typing import Any


class ClassifierModule(BaseLitModule):
    def __init__(self,
                 cfg:DictConfig,
                 *args:Any,
                 **kwargs:Any):
        self.network = cfg.network
        self.optimizer = cfg.optimizer
        self.scheduler = cfg.scheduler
        self.logging = cfg.logging
        self.metric = cfg.metric
        self.metric_name = cfg.metric['_target_'].split('.')[-1]

        super().__init__(self.network,self.optimizer,self.scheduler,self.logging,self.metric,*args,**kwargs)
        self.save_hyperparameters()
        self.logging = self.logging
        self.loss = hydra.utils.instantiate(cfg.loss)
    
    def _common_step(self,batch,stage):
        x,y = batch
        y_hat = self(x)
        loss = self.loss(y_hat,y)
        loss = loss + self.model.get_regularization_loss()
        metric = self.metric[stage](y_hat,y)
        return loss,metric,y_hat
    
    def training_step(self,batch,batch_idx):
        loss,metric,y_pred = self._common_step(batch,"train")
        self.log_dict({"train_loss":loss,f"train_{self.metric_name}":metric},**self.logging)
        return loss
    
    def validation_step(self,batch,batch_idx):
        loss,metric,y_pred = self._common_step(batch,"val")
        self.log_dict({"val_loss":loss,f"val_{self.metric_name}":metric},**self.logging)
        return loss
    
    def test_step(self,batch,batch_idx):
        loss,metric,y_pred = self._common_step(batch,"test")
        self.log_dict({"test_loss":loss,f"test_{self.metric_name}":metric},**self.logging)
        self.metric["test"]
        return loss

    

