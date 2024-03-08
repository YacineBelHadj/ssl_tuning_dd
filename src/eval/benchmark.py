
import torch
from torch import nn
from src.downstream.base_dds import BaseDDS
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from typing import List
import pandas as pd
from src.eval.utils import get_transform_fn, calculate_auc, get_anomaly_score
from scipy.stats import hmean
from omegaconf import ListConfig
from typing import Optional
from abc import ABC, abstractmethod 

# create an abstract class for the benchmark
class Benchmark_abstract(ABC):
    @abstractmethod
    def setup_loader(self):
        pass
    @abstractmethod
    def setup_anomaly_score(self):
        pass
    @abstractmethod
    def evaluate_individu(self):
        pass
    @abstractmethod
    def evaluate_all(self):
        pass

class Benchmark(Benchmark_abstract):
    """
    Class for evaluating the Virtual Anomaly Score (VAS) benchmark.

    Args:
        dds (BaseDDS): The base anomaly detection system contain predict that handler dataloader as input 
        and gives anomaly index.
        before_virtual_anomaly (DataLoader, optional): DataLoader for the data before the virtual anomaly. Defaults to None.
        after_virtual_anomaly (DataLoader, optional): DataLoader for the data after the virtual anomaly. Defaults to None.
        datamodule (LightningDataModule, optional): LightningDataModule for the data. Defaults to None.

    Raises:
        ValueError: If none of the data sources are provided.
        ValueError: If both DataLoader arguments are provided without datamodule, or datamodule is provided alone.

    Attributes:
        dds (BaseDDS): The base data source.
        before_virtual_anomaly (DataLoader): DataLoader for the data before the virtual anomaly.
        after_virtual_anomaly (DataLoader): DataLoader for the data after the virtual anomaly.

    Methods:
        setup_loader: Sets up the data loaders based on the provided arguments.
        compute_anomaly_score: Computes the anomaly scores for the given columns.
        compute_auc: Computes the Area Under the Curve (AUC) for the combined DataFrame.

    """

    def __init__(self, 
                 dds:BaseDDS,
                 dl1:DataLoader=None,
                 dl2:DataLoader=None,
                 datamodule:LightningDataModule=None,
                dl1_name:Optional[str]=None,
                dl2_name:Optional[str]=None,
                dl1_columns:List[str]=None,
                dl2_columns:List[str]=None,
                hue:str=None,
                anomaly_config:List[str]=None,
                transform_name:str=None):
        self.dds = dds
        self.setup_loader(dl1,dl2,datamodule,
                          dl1_name,dl2_name)
        assert set(dl1_columns).issubset(set(dl2_columns)), "All columns in before_anomaly should be present in after_anomaly"
        self.dl1_columns = dl1_columns
        self.dl2_columns = dl2_columns
        self.hue = hue
        self.anomaly_config = anomaly_config
        self.transform_fn = None
        self.is_setup = False
        self.setup_fn(transform_name)

    def setup_fn(self,transform_name:str=None):
        self.transform_fn = None

        if transform_name is not None:
            self.transform_fn = get_transform_fn(transform_name)

    def setup_loader(self, dl1, dl2, datamodule,dl1_name,dl2_name):
        """
        Sets up the data loaders based on the provided arguments.

        Args:
            before_virtual_anomaly (DataLoader): DataLoader for the data before the virtual anomaly.
            after_virtual_anomaly (DataLoader): DataLoader for the data after the virtual anomaly.
            datamodule (LightningDataModule): LightningDataModule for the data.

        Raises:
            ValueError: If none of the data sources are provided.
            ValueError: If both DataLoader arguments are provided without datamodule, or datamodule is provided alone.

        """

        
        # if datamodule is provided, and also before_virtual_anomaly and after_virtual_anomaly then raise an error
        if datamodule is not None and (dl1 is not None or dl2 is not None):
            raise ValueError("If datamodule is provided, dl1 and dl2 should not be provided.")
        # if datamodule is not provided, then before_virtual_anomaly and after_virtual_anomaly should be provided
        if datamodule is None and (dl1 is None or dl2 is None):
            raise ValueError("If datamodule is not provided, both dl's should be provided.")
        
        # Use datamodule if it's provided
        if datamodule is not None:
            # run the method to get the dataloader from the lit datamodule using the 
            # string name that is given 
            self.dl1 = datamodule.__getattribute__(dl1_name)()
            self.dl2 = datamodule.__getattribute__(dl2_name)()
        else:  # Use DataLoader arguments if datamodule is not provided
            self.dl1 = dl1
            self.dl2 = dl2

    def setup_anomaly_score(self):
        """
        Computes the anomaly scores and saves the columns 

        Args:
            before_anomaly_columns (List[str]): List of columns for the data before the virtual anomaly.
            after_anomaly_columns (List[str]): List of columns for the data after the virtual anomaly.

        Returns:
            pd.DataFrame: Combined DataFrame containing the anomaly scores.

        Raises:
            AssertionError: If not all columns in before_anomaly_columns are present in after_anomaly_columns.

        """
        # check that before_anomaly_column is contained in after_anomaly_column
        if self.is_setup:
            print("The anomaly scores are already set up.")
            pass
        dl1_score = get_anomaly_score(dds=self.dds, columns=self.dl1_columns, dataloader=self.dl1)
        dl2_score = get_anomaly_score(dds=self.dds,columns=self.dl2_columns,dataloader=self.dl2)

        # to pandas
        before_df = pd.DataFrame(dl1_score)
        after_df = pd.DataFrame(dl2_score)

        before_df['condition']='before'
        after_df['condition']='after'
        combined_df = pd.concat([before_df, after_df], ignore_index=True)
        self.combined_df = combined_df  
        if self.transform_fn is not None:
            self.combined_df['anomaly_index'] = self.transform_fn(self.combined_df['anomaly_index']) 
        self.is_setup = True

    def evaluate_individu(self,df_group):
        assert self.is_setup, "The anomaly scores should be set up first."
        
        # Work on a copy to avoid modifying the original DataFrame
        df_eval = df_group.copy()
        

        auc_res = []
        before_scores = df_eval[df_eval['condition'] == 'before']['anomaly_index'].values


        
        # Ensure grouping by anomaly_conf does not lead to empty groups
        if not all(col in df_eval.columns for col in self.anomaly_config):
            raise ValueError("One or more specified anomaly_config columns do not exist in df_group.")

        if isinstance(self.anomaly_config, ListConfig):
            self.anomaly_config = list(self.anomaly_config)   
        grouped_by_anomaly_config = df_eval[df_eval['condition'] == 'after'].groupby(self.anomaly_config)
        
        for config, group in grouped_by_anomaly_config:
            after_scores = group['anomaly_index'].values
            auc, sat = calculate_auc(before_scores, after_scores, saturation=True)
            
            config = config if isinstance(config, tuple) else (config,)
            config_dict = dict(zip(self.anomaly_config, config))
            temp = {**config_dict, 'auc': auc, 'saturation_applied': sat}
            auc_res.append(temp)

        # get harmonic mean of auc for all configurations
        # add the before_score with config values replaces by before 
        hmean_auc = hmean([res['auc'] for res in auc_res])
        return auc_res, hmean_auc
    
    def evaluate_all(self,**kwargs):
        assert self.is_setup, "The anomaly scores should be set up first."
        # Ensure hue is present in the DataFrame
        if self.hue not in self.combined_df.columns:
            raise ValueError(f"{self.hue} is not present in the combined DataFrame.")


        if not all(col in self.combined_df.columns for col in self.anomaly_config):
            raise ValueError("One or more specified anomaly_config columns do not exist in the combined DataFrame.")

        auc_all = []
        hmean_aucs = []
        grouped_by_hue = self.combined_df.groupby(self.hue)
        for individu, group in grouped_by_hue:
            auc_res, hmean_auc = self.evaluate_individu(group)
            hmean_aucs.append(hmean_auc)
            auc_all.append({self.hue: individu, 'auc': hmean_auc, 'individual_auc': auc_res})  
        hm_score = hmean(hmean_aucs)  
        return hm_score, auc_all
    


    



