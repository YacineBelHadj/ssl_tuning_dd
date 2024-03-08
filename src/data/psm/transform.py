from typing import Union
from pathlib import Path
import sqlite3
import numpy as np
import torch
import matplotlib.pyplot as plt

class CreateTransformer:
    def __init__(self, database_path: Union[str, Path], freq_min: int, freq_max: int, num_classes: int):
        self.database_path = database_path
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_classes = num_classes

        self.freq_before, self.min_psd, self.max_psd, self.dim_psd = self._initialize_params()

    def _initialize_params(self):
        with sqlite3.connect(self.database_path) as conn:
            c = conn.cursor()
            c.execute("SELECT freq FROM metadata")
            freq_before = np.frombuffer(c.fetchone()[0], dtype=np.float32)
            self.mask_cut_psd = (freq_before >= self.freq_min) & (freq_before <= self.freq_max)

            c.execute("SELECT PSD FROM processed_data WHERE stage='train'")
            psd_data = np.array([np.frombuffer(row[0],dtype=np.float32)[self.mask_cut_psd] for row in c.fetchall()])
            psd_data = np.log(psd_data)
            return freq_before, psd_data.min(), psd_data.max(), psd_data.shape[1]

    def get_transformer_psd(self):

        def transform_psd(psd):
            psd = np.frombuffer(psd, dtype=np.float32)
            psd = psd[self.mask_cut_psd]
            psd = np.log(psd)
            psd = (psd - self.min_psd) / (self.max_psd - self.min_psd)
            psd = torch.from_numpy(psd).float()
            return psd
        
        return transform_psd

    def get_transformer_label(self):
        def transform_label(label):
            l = int(label.split('_')[-1])
            return l
        return transform_label

        

    
# let's wrap this in a function
def create_psd_transformer(database_path, freq_min, freq_max, num_classes):
    transformer = CreateTransformer(database_path, freq_min, freq_max, num_classes)
    return transformer.get_transformer_psd()

def create_label_transformer(database_path, freq_min, freq_max, num_classes):
    transformer = CreateTransformer(database_path, freq_min, freq_max, num_classes)
    return transformer.get_transformer_label()    
    
def dimension_psd(database_path, freq_min, freq_max, num_classes):

    transformer = CreateTransformer(database_path, freq_min, freq_max, num_classes)
    print(transformer.dim_psd)
    return transformer.dim_psd


