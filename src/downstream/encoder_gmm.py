from .base_dds import BaseDDS
from typing import Any, TypeVar, Type, Union
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
        """
        Initializes an instance of the EncoderGMM class.

        Args:
            encoder (nn.modules): The encoder model.
            num_classes (int): The number of classes.

        Returns:
            None
        """
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        self.gm = GaussianMixture(n_components=num_classes, covariance_type='full')

    def load_embedding(self,loader:DataLoader) -> Any:
        """
        Loads the embeddings from the given data loader.

        Args:
            loader (DataLoader): The data loader.

        Returns:
            Any: The concatenated embeddings.
        """
        embeddings = []
        for batch in loader:
            with torch.no_grad():
                x , _ = batch
                embedding = self.encoder(x)
                embedding = embedding.detach().numpy()
                embeddings.append(embedding)
        return np.concatenate(embeddings)
    
    def fit(self,loader:DataLoader) -> None:
        """
        Fits the Gaussian Mixture Model (GMM) using the embeddings from the given data loader.

        Args:
            loader (DataLoader): The data loader.

        Returns:
            None
        """
        self.state = "fit"
        embedding = self.load_embedding(loader)
        self.gm.fit(embedding)

    def predict_dataloader(self,loader:DataLoader) -> Any:
        """
        Predicts the labels using the fitted GMM and the embeddings from the given data loader.

        Args:
            loader (DataLoader): The data loader.

        Returns:
            Any: The concatenated predicted labels.
        
        Raises:
            ValueError: If the model is not fit yet.
        """
        if self.state != "fit":
            raise ValueError("Model not fit yet")
        labels = []
        for batch in loader:
            with torch.no_grad():
                x  = batch
                x = x.detach().numpy()
                embedding = self.encoder(x)

                # get likelihood of the embedding
                label = self.gm.score_samples(embedding)
                labels.append(label)
        return np.concatenate(labels,dim=0)
    
    def predict_samples(self,sample:np.ndarray) -> Any:
        """
        Predicts the labels using the fitted GMM and the given samples.

        Args:
            sample (np.ndarray): The samples.

        Returns:
            Any: The predicted labels.
        
        Raises:
            ValueError: If the model is not fit yet.
        """
        if self.state != "fit":
            raise ValueError("Model not fit yet")
        if len(sample.shape) == 1:
            sample = sample.reshape(1,-1)

        with torch.no_grad():
            embedding = self.encoder(sample)
            label = self.gm.score_samples(embedding)
        return label

    # add a function to make the object callable and return the predict function
    def __call__(self,data:Union[DataLoader,np.ndarray]) -> Any:
        """
        Predicts the labels using the fitted GMM and the given data loader or samples.

        Args:
            loader (Union[DataLoader,np.ndarray]): The data loader or samples.

        Returns:
            Any: The predicted labels.
        
        Raises:
            ValueError: If the model is not fit yet.
        """
        if isinstance(data,DataLoader):
            return self.predict_dataloader(data)
        elif isinstance(data,np.ndarray) or isinstance(data,torch.Tensor):
            return self.predict_samples(data)
        else:
            raise ValueError(f"Input should be a DataLoader \
                             or a numpy array or torchtensor, but instead got {type(data)}")
    
    def save(self,path:str) -> None:
        """
        Saves the encoder and GMM to the specified path.

        Args:
            path (str): The path to save the encoder and GMM.

        Returns:
            None
        """
        directory = Path(path)
        directory.mkdir(parents=True,exist_ok=True)
        encoder_path = Path(path)/"encoder.pth"
        gm_path = Path(path)/"gm.joblib"
        torch.save(self.encoder.state_dict(),encoder_path)
        joblib.dump(self.gm,gm_path)
    
    @classmethod
    def load(cls: Type[M], path: str, encoder: nn.Module, num_classes: int) -> M:
        """
        Loads the encoder and GMM from the specified path and returns an instance of the EncoderGMM class.

        Args:
            cls (Type[M]): The class type.
            path (str): The path to load the encoder and GMM.
            encoder (nn.Module): The encoder model.
            num_classes (int): The number of classes.

        Returns:
            M: An instance of the EncoderGMM class.

        """
        instance = cls(encoder, num_classes)
        encoder_path = Path(path) / "encoder.pth"
        gm_path = Path(path) / "gm.joblib"
        instance.encoder.load_state_dict(torch.load(encoder_path))
        instance.gm = joblib.load(gm_path)
        instance.state = "fit"  # Assuming loading implies the model was previously fit
        return instance
        

    

    

    
        
