from model.lit_module import BaseLitModule
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
        self.log_dict({"train_loss":loss,"train_metric":metric})
        return loss
    
    def validation_step(self,batch,batch_idx):
        loss,metric,y_pred = self._common_step(batch,"val")
        self.log_dict({"val_loss":loss,"val_metric":metric})
        return loss
    
    def test_step(self,batch,batch_idx):
        loss,metric,y_pred = self._common_step(batch,"test")
        self.log_dict({"test_loss":loss,"test_metric":metric})
        self.metric["test"]
        return loss

    

