import matplotlib.pyplot as plt
import numpy as np 
from typing import List, Callable
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from omegaconf import DictConfig, ListConfig

def plot_psd_fn():
    def wrap(x:torch.Tensor,ax=None):
        """
        Method to plot the PSD of the given instance.

        Args:
            x (torch.Tensor): Instance to plot.
            ax (Any, optional): Axes to plot on. Defaults to None.

        Returns:
            Any: Axes object.
        """

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(x)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('PSD')
        return ax
    return wrap


class DatamoduleVis:
    def __init__(self,datamodule:LightningDataModule,number_of_instances:int=6,
                 dataset_name:List[str]=['train','val','test','structural_anomaly','before_virtual_anomaly','after_virtual_anomaly'],
                 ploting_func:Callable=plot_psd_fn, verbose:bool=True):
        
        self.datamodule = datamodule
        self.number_of_instances = number_of_instances
        self.verbose = verbose
        self.dataset_name = dataset_name
        if isinstance(self.dataset_name,ListConfig):
            self.dataset_name = list(self.dataset_name)
        print(ploting_func)
        self.ploting_func = ploting_func()
        print(self.ploting_func)
        if self.verbose:
            self.visualize()


    
    def plot_dataset(self,dataset:DataLoader,title:str='Dataset'):
        """
        Method to plot the given dataset.

        Args:
            dataset (DataLoader): Dataset to plot.
        """
        #create a grid of subplots  with the number of instances
        width = int(np.sqrt(self.number_of_instances))
        height = int(np.ceil(self.number_of_instances/width))
        fig_size=(width*5,height*5)
        fig, axs = plt.subplots(height, width,figsize=fig_size)
        for k in range(self.number_of_instances):
            #get a random instance from the dataset
            i = np.random.randint(len(dataset))
            x,*rest = dataset[i]
            ax = axs[k//width,k%width]
            self.ploting_func(x,ax) 
            ax.set_title(f'Instance {i} || Label: {rest}')
        # set the title of the figure
        fig.suptitle(title)
        return fig,axs
        

    def visualize(self):
        axs_dict = {}
        for name in self.dataset_name:
            dataset = getattr(self.datamodule,name+'_ds')
            fig,axs= self.plot_dataset(dataset,title=name)
            #visualize the axes
            plt.show()
            axs_dict[name] = axs
            plt.close(fig)
            #wait 5 seconds and close the figure automatically
            
        return axs_dict



           
if __name__ == "__main__":
    number_of_instances = 15
    width = int(np.sqrt(number_of_instances))
    height = int(np.ceil(number_of_instances/width))
    print(width,height)