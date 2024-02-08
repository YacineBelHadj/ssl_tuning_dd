from .base_dds import BaseDDS
from typing import Any, TypeVar, Type
from sklearn.mixture import GaussianMixture
from torch import nn
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import joblib

M = TypeVar("M", bound="EncoderGMM")

class EncoderGMM(BaseDDS):
    def __init__(self,encoder:nn.modules,num_classes:int) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        self.gm = GaussianMixture(n_components=num_classes, covariance_type='full')

    def load_embedding(self,loader:DataLoader) -> Any:
        embeddings = []
        for batch in loader:
            with torch.no_grad():
                x , _ = batch
                x = x.detach().numpy()
                embedding = self.encoder(x)
                embeddings.append(embedding)
        return np.concatenate(embeddings,dim=0)
    
    def fit(self,loader:DataLoader) -> None:
        self.state = "fit"
        embedding = self.load_embedding(loader)
        self.gm.fit(embedding)

    def predict(self,loader:DataLoader) -> Any:
        if self.state != "fit":
            raise ValueError("Model not fit yet")
        labels = []
        for batch in loader:
            with torch.no_grad():
                x  = batch
                x = x.detach().numpy()
                embedding = self.encoder(x)
                label = self.gm.predict(embedding)
                labels.append(label)
        return np.concatenate(labels,dim=0)
    
    def save(self,path:str) -> None:
        directory = Path(path)
        directory.mkdir(parents=True,exist_ok=True)
        encoder_path = Path(path)/"encoder.pth"
        gm_path = Path(path)/"gm.joblib"
        torch.save(self.encoder.state_dict(),encoder_path)
        joblib.dump(self.gm,gm_path)
        

    

    

    
        
