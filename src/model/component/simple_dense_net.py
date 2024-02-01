from torch import nn 
from typing import List
import torch
from typing import Union
from src.component.losses.regularization import l1_regularization,l2_regularization

class SimpleDenseNet(nn.Module):
    def __init__(
            self,
            input_dim:Union[int,str],
            hidden_layers:List[int],
            embedding_dim:int,
            output_dim:int,
            dropout:float,
            activation:str,
            batch_norm:bool,
            bias:bool,
            temperature:float,
            l1_regularization:float,
            l2_regularization:float,
            ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_layers = hidden_layers
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = getattr(nn,activation)
        self.batch_norm = batch_norm
        self.bias = bias
        self.temperature = temperature
        self.l1_regularization = l1_regularization
        self.l2_regularization = l2_regularization
        self.build()

    def build(self):
        all_layers = self.hidden_layers +[self.embedding_dim]
        self.layers = nn.ModuleList()
        in_feature = self.input_dim
        for out_feature in all_layers:
            self.layers.append(nn.Linear(in_feature,out_feature,bias=self.bias))
            if self.batch_norm:
                self.layers.append(nn.BatchNorm1d(out_feature))
            self.layers.append(self.activation())
            self.layers.append(nn.Dropout(self.dropout))
            in_feature = out_feature
        
        self.encoder = nn.Sequential(*self.layers)
        self.logit_layer= nn.Linear(in_feature,self.output_dim,bias=self.bias)
        self.full_model = nn.Sequential(self.encoder,self.logit_layer)

    def forward(self,x):
        embedding= self.encoder(x)
        x = self.logit_layer(embedding)
        return x
    
    def get_regularization_loss(self):
        l1_loss = l1_regularization(self.full_model)*self.l1_regularization
        l2_loss = l2_regularization(self.full_model)*self.l2_regularization
        return l1_loss+l2_loss
    






