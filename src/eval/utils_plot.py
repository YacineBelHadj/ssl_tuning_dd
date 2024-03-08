
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List

def plot_heatmap(data:pd.DataFrame,ax=None,axis_label=['f_anomaly,a_anomaly'],title='system_uk'):
    """
    Function to plot a heatmap of the given data.

    Args:
        data (pd.DataFrame): The data to plot.
        title (str, optional): The title of the plot. Defaults to 'Heatmap'.
        figsize (tuple, optional): The size of the figure. Defaults to (10,5).

    """
    xaxis = axis_label[0]
    yaxis = axis_label[1]
    
    if isinstance(data,List):
        data = pd.DataFrame(data)
    if 'auc' not in data.columns:
        raise ValueError("The data should have a column named 'auc'.")
    
    data = data.pivot(index=yaxis, columns=xaxis, values='auc')
    if ax is None:
        fig, ax = plt.subplots()

    im = ax.imshow(data, cmap='hot', interpolation='nearest', aspect='auto')

    # Set x and y ticks
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_yticks(np.arange(len(data.index)))
    # Set x and y tick labels
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.index)

    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Show color bar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("AUC", rotation=-90, va="bottom")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.set_title(title)

    return fig, ax 

def plot_boxplot(data:pd.DataFrame,ax=None,axis_label=['stiffness_reduction','anomaly_index'],title='system_uk'):
    xaxis = axis_label[0]
    yaxis = axis_label[1]
    if ax is None:
        fig, ax = plt.subplots()

    data.boxplot(column=yaxis, by=xaxis, ax=ax,showfliers=False)
    ax.set_title(title)
    ax.set_yscale('log')
    return fig, ax


    
