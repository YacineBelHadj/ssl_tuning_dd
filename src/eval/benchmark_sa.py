
from src.downstream.base_dds import BaseDDS
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from typing import List
from src.eval.utils import get_anomaly_score
import pandas as pd

class Benchmark_SA:
    """ 
    Class for evaluation th structural anomaly
    Args (BaseDDS): the """

    def __init__(self,
                 dds:BaseDDS,
                 test:DataLoader=None,
                 structural_anomaly:DataLoader=None,
                 datamodule:LightningDataModule=None):
        
        self.dds = dds
        self.setup_loader(test,structural_anomaly,datamodule)
        self.is_setupted = False

    def setup_loader(self,test,structural_anomaly,datamodule):
        if datamodule is not None and (test is not None or structural_anomaly is not None):
            raise ValueError("If datamodule is provided, before_virtual_anomaly and after_virtual_anomaly should not be provided.")
        # if datamodule is not provided, then before_virtual_anomaly and after_virtual_anomaly should be provided
        if datamodule is None and (test is None or structural_anomaly is None):
            raise ValueError("If datamodule is not provided, both before_virtual_anomaly and after_virtual_anomaly should be provided.")
        
        # Use datamodule if it's provided
        if datamodule is not None:
            self.test = datamodule.test_dataloader()
            self.after_virtual_anomaly = datamodule.structural_anomaly_dataloader()
        else:  # Use DataLoader arguments if datamodule is not provided
            self.test = test
            self.after_virtual_anomaly = structural_anomaly

    def setup_anomaly_score(self,test_columns:List[str],structural_anomaly_columns:List[str]):
        """
        Computes the anomaly scores for the given columns.
        """
        assert set(test_columns).issubset(set(structural_anomaly_columns)), "The columns should be the same"
        test_scores = get_anomaly_score(dds=self.dds,columns=test_columns,dataloader=self.test)
        structural_anomaly_scores = get_anomaly_score(dds=self.dds,columns=structural_anomaly_columns,
                                                      dataloader=self.after_virtual_anomaly)
        
        before_df = pd.DataFrame(test_scores)
        after_df = pd.DataFrame(structural_anomaly_scores)

        before_df["condition"] = "before"
        after_df["condition"] = "after"
        combined_df = pd.concat([before_df, after_df], ignore_index=True)
        self.combined_df = combined_df
        self.is_setupted = True

    def evaluate_individu(self,
                          df_group, transform_fn: callable = None,
                          a)

        

