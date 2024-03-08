import pytorch_lightning as pl
import torch
import hydra
from torch.utils.data import DataLoader,random_split
from omegaconf import DictConfig, OmegaConf

class DataModule(pl.LightningDataModule):
    """
    LightningDataModule subclass for handling data loading and processing in the PSM package.

    Args:
        train_val (DictConfig): Configuration for the training and validation dataset.
        test (DictConfig): Configuration for the test dataset.
        structural_anomaly (DictConfig): Configuration for the structural anomaly dataset.
        before_virtual_anomaly (DictConfig): Configuration for the dataset before virtual anomaly.
        after_virtual_anomaly (DictConfig): Configuration for the dataset after virtual anomaly.
    """

    def __init__(self,
                 train_val:DictConfig,
                 test:DictConfig,
                 structural_anomaly:DictConfig,
                 before_virtual_anomaly:DictConfig,
                 after_virtual_anomaly:DictConfig):
        super().__init__()
        self.train_val = train_val
        self.test = test
        self.structural_anomaly = structural_anomaly
        self.before_virtual_anomaly = before_virtual_anomaly
        self.after_virtual_anomaly = after_virtual_anomaly

    def _get_dataset(self,dataset_config):
        """
        Helper method to instantiate a dataset based on the provided configuration.

        Args:
            dataset_config (DictConfig): Configuration for the dataset.

        Returns:
            Dataset: Instantiated dataset object.
        """
        return hydra.utils.instantiate(dataset_config.dataset)
    
    def setup(self,stage=None):
        """
        Method to set up the data module.

        Args:
            stage (str, optional): Stage of setup. Defaults to None.
        """
        self.train_val_ds = self._get_dataset(self.train_val)
        self.test_ds = self._get_dataset(self.test)
        self.structural_anomaly_ds = self._get_dataset(self.structural_anomaly)
        self.before_virtual_anomaly_ds = self._get_dataset(self.before_virtual_anomaly)
        self.after_virtual_anomaly_ds = self._get_dataset(self.after_virtual_anomaly)
        total_length = len(self.train_val_ds)
        self.len_val = int(total_length * 0.2)
        self.len_train = total_length - self.len_val
        self.train_ds,self.val_ds = random_split(self.train_val_ds,[self.len_train,self.len_val])

    def remove_dataset_key(self,dataset_config):
        """
        Helper method to remove the 'dataset' key from the dataset configuration.

        Args:
            dataset_config (DictConfig): Configuration for the dataset.

        Returns:
            DictConfig: Configuration without the 'dataset' key.
        """
        new = dataset_config.copy()
        new.pop("dataset",None)
        return new
    
    def train_dataloader(self):
        """
        Method to get the training dataloader.

        Returns:
            DataLoader: Training dataloader.
        """
        p = self.remove_dataset_key(self.train_val)
        return DataLoader(self.train_ds,**p)
    
    def val_dataloader(self):
        """
        Method to get the validation dataloader.

        Returns:
            DataLoader: Validation dataloader.
        """
        param = self.train_val
        param['shuffle'] = False
        #drop dataset key
        p = self.remove_dataset_key(param)
        return DataLoader(self.val_ds,**p)
    
    def test_dataloader(self):
        """
        Method to get the test dataloader.

        Returns:
            DataLoader: Test dataloader.
        """
        p = self.remove_dataset_key(self.test)
        return DataLoader(self.test_ds,**p)
    
    def structural_anomaly_dataloader(self):
        """
        Method to get the dataloader for the structural anomaly dataset.

        Returns:
            DataLoader: Dataloader for the structural anomaly dataset.
        """
        p = self.remove_dataset_key(self.structural_anomaly)
        return DataLoader(self.structural_anomaly_ds,**p)
    
    def before_virtual_anomaly_dataloader(self):
        """
        Method to get the dataloader for the dataset before virtual anomaly.

        Returns:
            DataLoader: Dataloader for the dataset before virtual anomaly.
        """
        p = self.remove_dataset_key(self.before_virtual_anomaly)
        return DataLoader(self.before_virtual_anomaly_ds,**p)
    
    def after_virtual_anomaly_dataloader(self):
        """
        Method to get the dataloader for the dataset after virtual anomaly.

        Returns:
            DataLoader: Dataloader for the dataset after virtual anomaly.
        """
        p = self.remove_dataset_key(self.after_virtual_anomaly)
        return DataLoader(self.after_virtual_anomaly_ds,**p)
    


            
        