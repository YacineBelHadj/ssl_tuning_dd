
import torch
from torch import nn
from src.downstream.base_dds import BaseDDS
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from typing import List
from src.eval.utils import get_anomaly_score, _calculate_auc_hue
import pandas as pd
from src.eval.utils import inverse, calculate_auc
from scipy.stats import hmean
from src.utils.rich_utils import with_progress_bar
from omegaconf import ListConfig
class Benchmark_VAS:
    """
    Class for evaluating the Virtual Anomaly Score (VAS) benchmark.

    Args:
        dds (BaseDDS): The base data source.
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
                 before_virtual_anomaly:DataLoader=None,
                 after_virtual_anomaly:DataLoader=None,
                 datamodule:LightningDataModule=None):
        self.dds = dds
        self.setup_loader(before_virtual_anomaly,
                          after_virtual_anomaly,datamodule)

    def setup_loader(self, before_virtual_anomaly, after_virtual_anomaly, datamodule):
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
        if datamodule is not None and (before_virtual_anomaly is not None or after_virtual_anomaly is not None):
            raise ValueError("If datamodule is provided, before_virtual_anomaly and after_virtual_anomaly should not be provided.")
        # if datamodule is not provided, then before_virtual_anomaly and after_virtual_anomaly should be provided
        if datamodule is None and (before_virtual_anomaly is None or after_virtual_anomaly is None):
            raise ValueError("If datamodule is not provided, both before_virtual_anomaly and after_virtual_anomaly should be provided.")
        
        # Use datamodule if it's provided
        if datamodule is not None:
            self.before_virtual_anomaly = datamodule.before_virtual_anomaly_dataloader()
            self.after_virtual_anomaly = datamodule.after_virtual_anomaly_dataloader()
        else:  # Use DataLoader arguments if datamodule is not provided
            self.before_virtual_anomaly = before_virtual_anomaly()
            self.after_virtual_anomaly = after_virtual_anomaly()

    def setup_anomaly_score(self,before_anomaly_columns:List[str],after_anomaly_columns:List[str]):
        """
        Computes the anomaly scores for the given columns.

        Args:
            before_anomaly_columns (List[str]): List of columns for the data before the virtual anomaly.
            after_anomaly_columns (List[str]): List of columns for the data after the virtual anomaly.

        Returns:
            pd.DataFrame: Combined DataFrame containing the anomaly scores.

        Raises:
            AssertionError: If not all columns in before_anomaly_columns are present in after_anomaly_columns.

        """
        # check that before_anomaly_column is contained in after_anomaly_column
        assert set(before_anomaly_columns).issubset(set(after_anomaly_columns)), "All columns in before_anomaly should be present in after_anomaly"
        before_anomaly_scores = get_anomaly_score(dds=self.dds, columns=before_anomaly_columns, dataloader=self.before_virtual_anomaly)
        after_anomaly_scores = get_anomaly_score(dds=self.dds,columns=after_anomaly_columns,dataloader=self.after_virtual_anomaly)

        # to pandas
        before_df = pd.DataFrame(before_anomaly_scores)
        after_df = pd.DataFrame(after_anomaly_scores)

        before_df['condition']='before'
        after_df['condition']='after'
        combined_df = pd.concat([before_df, after_df], ignore_index=True)
        self.combined_df = combined_df   
        self.is_setup = True

    def evaluate_individu(self,
                        df_group, transform_fn: callable = None, 
                          anomaly_config: List[str] = ['freq_a', 'amplitude_a']
                          ):
        assert self.is_setup, "The anomaly scores should be set up first."
        
        # Work on a copy to avoid modifying the original DataFrame
        df_eval = df_group.copy()
        
        # Apply transformation if provided
        if transform_fn:
            df_eval['anomaly_index'] = transform_fn(df_eval['anomaly_index'])
        else:
            raise ValueError("Provided transform is not callable.")

        auc_res = []
        before_scores = df_eval[df_eval['condition'] == 'before']['anomaly_index'].values
        
        # Ensure grouping by anomaly_conf does not lead to empty groups
        if not all(col in df_eval.columns for col in anomaly_config):
            raise ValueError("One or more specified anomaly_config columns do not exist in df_group.")

        if isinstance(anomaly_config, ListConfig):
            anomaly_config = list(anomaly_config)   
        grouped_by_anomaly_config = df_eval[df_eval['condition'] == 'after'].groupby(anomaly_config)
        
        for config, group in grouped_by_anomaly_config:
            after_scores = group['anomaly_index'].values
            auc, sat = calculate_auc(before_scores, after_scores, saturation=True)
            
            config = config if isinstance(config, tuple) else (config,)
            config_dict = dict(zip(anomaly_config, config))

            auc_res.append({**config_dict, 'auc': auc, 'saturation_applied': sat})
        # get harmonic mean of auc for all configurations
        hmean_auc = hmean([res['auc'] for res in auc_res])
        return auc_res, hmean_auc
    
    def evaluate_all(self, hue: str, transform: str='inverse',
                      anomaly_config: List[str] = ['f_anomaly', 'l_anomaly'],**kwargs):
        

        assert self.is_setup, "The anomaly scores should be set up first."

        # Ensure hue is present in the DataFrame
        if hue not in self.combined_df.columns:
            raise ValueError(f"{hue} is not present in the combined DataFrame.")

        # Ensure grouping by hue does not lead to empty groups
        print(self.combined_df.columns)
        print (anomaly_config)

        if not all(col in self.combined_df.columns for col in anomaly_config):
            raise ValueError("One or more specified anomaly_config columns do not exist in the combined DataFrame.")

        transform_fn = get_transform_fn(transform)
        auc_all = []
        hmean_aucs = []
        grouped_by_hue = self.combined_df.groupby(hue)
        for individu, group in grouped_by_hue:
            auc_res, hmean_auc = self.evaluate_individu(group, transform_fn, anomaly_config)
            hmean_aucs.append(hmean_auc)
            auc_all.append({hue: individu, 'auc': hmean_auc, 'individual_auc': auc_res})  
        vas = hmean(hmean_aucs)  

        return pd.DataFrame(auc_all), vas
    

def get_transform_fn(transform):
    if transform == 'inverse':
        return inverse
    else:
        raise ValueError(f"Unknown transform {transform}.")