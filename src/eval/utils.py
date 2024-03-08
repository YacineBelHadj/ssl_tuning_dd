from typing import List
from src.downstream.base_dds import BaseDDS
from torch.utils.data import DataLoader
import torch 
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from src.utils.rich_utils import with_progress_bar

##
@with_progress_bar
def get_anomaly_score(dds:BaseDDS, columns:List[str], dataloader:DataLoader,key_name:str="anomaly_index",**kwargs):
    #check if the dataloader has the same length of columns
    if len(columns)+1 != len(dataloader.dataset[0]):
        raise ValueError("The number of columns should be the same as the number of columns in the dataloader.")
    
 

    # make a dictionary to store the anomaly scores and the other columns
    res = {col:[] for col in columns}
    res[key_name] = []
    total_work = len(dataloader)

    progress = kwargs.get('progress')
    task_id = kwargs.get('task_id')
    # get the function name

    progress.update(task_id, total=total_work, description=f"[green]Executing task", advance=1)

    for batch in dataloader:
        psd,*data_columns = batch
        with torch.no_grad():
            anomaly_score = dds(psd)
        for i,col in enumerate(columns):
            res[col].extend(data_columns[i].tolist())
        
        res[key_name].extend(anomaly_score.tolist())
        progress.update(task_id, advance=1)
    return res

def calculate_auc(before: np.ndarray, after: np.ndarray, saturation: bool = False):
    """
    Computes the Area Under the Curve (AUC) for anomaly scores before and after a virtual anomaly.
    
    The function concatenates the before and after scores to calculate the AUC, 
    allowing an option to 'saturate' AUC values below a certain threshold to 0.5.
    
    Args:
        before (np.ndarray): Anomaly scores before the virtual anomaly.
        after (np.ndarray): Anomaly scores after the virtual anomaly.
        saturation (bool, optional): If True, AUC values below 0.4 are set to 0.5. Defaults to False.
    
    Returns:
        tuple: A tuple containing the calculated AUC value and a saturation flag (0 or 1).
    
    Raises:
        ValueError: If 'before' and 'after' arrays do not have the same length.
    """
    if len(before) != len(after):
        raise ValueError("The 'before' and 'after' arrays must have the same length.")
    
    # Combine the labels for before (0) and after (1) into a single array
    y_true = np.concatenate([np.zeros(len(before)), np.ones(len(after))])
    # Combine the scores from before and after into a single array
    y_score = np.concatenate([before, after])
    # Calculate AUC
    auc = roc_auc_score(y_true, y_score)

    # Apply saturation logic if enabled
    sat = 0  # Saturation flag, 0 by default indicating no saturation applied
    if saturation and auc < 0.4:
        auc = 0.5
        sat = 1  # Update flag to indicate saturation was applied

    if saturation:
        return auc, sat
    else:
        return auc

def calculate_auc_config(df:pd.DataFrame,anomaly_conf:List[str]):
    """
    Computes the AUC for different configurations of anomaly scores.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the anomaly scores.
    anomaly_conf (List[str]): The list of columns containing the anomaly scores.
    
    Returns:
    dict: A dictionary containing the AUC for each configuration.
    """
    auc_results_per_system = {}
    refernce = df[df['condition']=='before']
    for conf in anomaly_conf:
        auc = roc_auc_score(refernce['anomaly_index'],df[df['condition']=='after'][conf])
        auc_results_per_system[conf] = auc
    return auc_results_per_system

def _calculate_auc_hue(df:pd.DataFrame,hue:str,anomaly_conf=List[str],transform:callable=None):
        """
        Calculates AUC for each system separately based on the 'condition' column.
        
        Args:
        combined_df (pd.DataFrame): The combined DataFrame containing both before and after anomaly scores.
        hue (str): The column name used to group the data by system or scenario.
        
        Returns:
        pd.DataFrame: A DataFrame containing the AUC results for each system.
        """
        auc_results = []
        #check if columns anomaly_index and condition are present
        assert set(['condition','anomaly_index']).issubset(df.columns), \
        "calculate_auc_hue is called with a dataframe that does not have condition and anomaly_index in the columns"
        # assert that condition contains only two unique values before and after
        assert set(df['condition'].unique()) == set(['before','after']), \
        "calculate_auc_hue is called with a dataframe that does not have only two unique values in the condition column"

        for name, group in df.groupby(hue):
            if len(group['anomaly_index'].unique())==1:
                raise ValueError(f"Only one unique value in the anomaly_index column for {name}")
            if transform is not None:
                group['anomaly_index'] = transform(group['anomaly_index'])
            auc_list = calculate_auc_config(group,anomaly_conf=anomaly_conf)
            auc_results.append({hue:name,**auc_list})
                 
###

def inverse(anomaly_score):
    """
    Transforms the anomaly score to a value between 0 and 1.
    
    Args:
    anomaly_score (torch.Tensor): The anomaly score.
    
    Returns:
    """
    return -anomaly_score
    
def get_transform_fn(transform):
    if transform == 'inverse':
        return inverse
    else:
        raise ValueError(f"Unknown transform {transform}.")