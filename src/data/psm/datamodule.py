import pytorch_lightning as pl
import torch
import hydra
from torch.utils.data import DataLoader,random_split
from omegaconf import DictConfig, OmegaConf

class DataModule(pl.LightningDataModule):
    def __init__(self,
                 train_val:DictConfig,
                 test:DictConfig,
                 affected_real_anomaly:DictConfig,
                 reference:DictConfig,
                 affected_virtual_anomaly:DictConfig):
        super().__init__()
        self.train_val = train_val
        self.test = test
        self.affected_real_anomaly = affected_real_anomaly
        self.reference = reference
        self.affacted_virtual_anomaly = affected_virtual_anomaly

    def _get_dataset(self,dataset_config):
        return hydra.utils.instantiate(dataset_config.dataset)
    
    def setup(self,stage=None):
        self.train_val_ds = self._get_dataset(self.train_val)
        self.test_ds = self._get_dataset(self.test)
        self.affected_real_anomaly_ds = self._get_dataset(self.affected_real_anomaly)
        self.reference_ds = self._get_dataset(self.reference)
        self.affacted_virtual_anomaly_ds = self._get_dataset(self.affacted_virtual_anomaly)
        total_length = len(self.train_val_ds)
        self.len_val = int(total_length * 0.8)
        self.len_train = total_length - self.len_val
        self.train_ds,self.val_ds = random_split(self.train_val_ds,[self.len_train,self.len_val])

    def remove_dataset_key(self,dataset_config):
        new = dataset_config.copy()
        new.pop("dataset",None)
        return new
    
    def train_dataloader(self):
        p = self.remove_dataset_key(self.train_val)
        return DataLoader(self.train_ds,**p)
    
    def val_dataloader(self):
        param = self.train_val
        param['shuffle'] = False
        #drop dataset key
        p = self.remove_dataset_key(param)
        return DataLoader(self.val_ds,**p)
    
    def test_dataloader(self):
        p = self.remove_dataset_key(self.test)
        return DataLoader(self.test_ds,**p)
    
    def affected_real_anomaly_dataloader(self):
        p = self.remove_dataset_key(self.affected_real_anomaly)
        return DataLoader(self.affected_real_anomaly_ds,**p)
    
    def reference_dataloader(self):
        p = self.remove_dataset_key(self.reference)
        return DataLoader(self.reference_ds,**p)
    
    def affacted_virtual_anomaly_dataloader(self):
        p = self.remove_dataset_key(self.affacted_virtual_anomaly)
        return DataLoader(self.affacted_virtual_anomaly_ds,**p)
    


            
        