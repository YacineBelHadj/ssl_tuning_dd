from typing import Union
from pathlib import Path
import sqlite3
import numpy as np

class CreateTransformer:
    """ This class is used to create a transformer for the data module.
    it transform the psd with min max scaler 
    it also filter the psd with a band pass filter 
    it apply a log transform to the psd
    >>> transformer = CreateTransformer(database_path,freq_min,freq_max)
    >>> transform_psd = transformer.get_transformer_psd()
    >>> transform_label = transformer.get_transformer_label()
    """

    def __init__(self,database_path: Union[str, Path],freq_min: int,freq_max: int,num_classes: int):
        self.database_path = database_path
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_classes = num_classes
        

    @property  
    def freq_before(self):
        """ return the freq axis of the psd """
        conn = sqlite3.connect(self.database_path)
        c = conn.cursor()
        c.execute("SELECT freq FROM metadata") # this is in Bytes
        freq_before = np.frombuffer(c.fetchone()[0])
        conn.close()
        return freq_before
    
    @property
    def mask_cut_psd(self):
        mask_cut_psd = (self.freq_before >= self.freq_min) & (self.freq_before <= self.freq_max)
        return mask_cut_psd
    
    @property
    def freq_after(self):
        mask = self.mask_cut_psd()
        freq_after = self.freq_before[mask]
        return freq_after
    

    def cut_psd(self, psd):
        """ Cut the psd to the freq_min and freq_max. """
        mask = self.mask_cut_psd
        if psd.ndim == 1:
            return psd[mask]
        elif psd.ndim == 2:
            return psd[:,mask]
        else:
            raise ValueError("psd should be 1 or 2 dim")
            
    def min_max_dim(self):
        conn = sqlite3.connect(self.database_path)
        c = conn.cursor()
        # in the processed table the psd is stored as a blob and we want to take the min and max of the psd where stage ='train'
        c.execute("SELECT PSD FROM processed_data WHERE stage='train'")      
         
        psd = np.array([np.frombuffer(row[0]) for row in c.fetchall()])
        psd= self.cut_psd(psd)
        psd = np.log(psd)
        conn.close()
        dim_psd = psd.shape[1]
        return psd.min(), psd.max(), dim_psd
    
    @property
    def min_psd(self):
        return self.min_max_dim()[0]
    @property
    def max_psd(self):
        return self.min_max_dim()[1]
    @property
    def dim_psd(self):
        return self.min_max_dim()[2]
    @property
    def transform_psd(self):
        """ return the transformer for the psd """
        def func(psd):
            psd = self.cut_psd(psd)
            psd = np.log(psd)
            psd = (psd - self.min_psd) / (self.max_psd - self.min_psd)
            return psd
        return func
    @property  
    def transform_label(self):
        """ return the transformer for the label """
        def func(x):
            l = int(x.split('_')[-1])
            print(l)
            # one-hot encoding
            return l
        return func
        

    
# let's wrap this in a function
def create_psd_transformer(database_path, freq_min, freq_max, num_classes):
    transformer = CreateTransformer(database_path, freq_min, freq_max, num_classes)
    print('hy')
    return transformer.transform_psd

def create_label_transformer(database_path, freq_min, freq_max, num_classes):
    transformer = CreateTransformer(database_path, freq_min, freq_max, num_classes)
    return transformer.transform_label    
    
def dimension_psd(database_path, freq_min, freq_max, num_classes):
    print('aa')

    transformer = CreateTransformer(database_path, freq_min, freq_max, num_classes)
    return transformer.dim_psd


