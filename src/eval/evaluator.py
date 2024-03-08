
from src.downstream.base_dds import BaseDDS
from pytorch_lightning import LightningDataModule
from typing import List
from omegaconf import  ListConfig
from pytorch_lightning.loggers import Logger
import hydra
import matplotlib as mpl
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
from src.eval.utils_plot import plot_boxplot , plot_heatmap

mpl.use('TkAgg')

class Evaluator:
    def __init__(self,cfg,dds:BaseDDS,datamodule:LightningDataModule,logger:Logger):
        self.benchmark_vas = hydra.utils.instantiate(cfg.benchmark_vas,dds=dds,datamodule=datamodule)
        self.benchmark_sa = hydra.utils.instantiate(cfg.benchmark_sa,dds=dds,datamodule=datamodule)
        self.logger = logger[0]
        self.hue = cfg.benchmark_sa.hue

    def setup(self):
        self.benchmark_vas.setup_anomaly_score()
        self.benchmark_sa.setup_anomaly_score()
        
        
    def evaluate_vas(self):
        score_vas, auc_all_vas = self.benchmark_vas.evaluate_all()
        # log the plots that are saves in axs_dict_vas 
        df= pd.DataFrame(auc_all_vas)
        # iterate over each row 
        for _,rows in df.iterrows(): 
            title_name = f"System {rows.system_name}: AUC = {rows.auc:.3f}" 
            data = pd.DataFrame(rows.individual_auc)[['f_anomaly','a_anomaly','auc']]
            fig,ax=plot_heatmap(data,axis_label=self.benchmark_vas.anomaly_config,title=title_name)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            self.logger.experiment.log_image(buf, name=title_name)
            plt.close(fig)
            buf.close()
            self.logger.log_metrics({f'VAS_System_{rows.system_name}':rows.auc})
        self.logger.log_metrics({'VAS':score_vas})
        return score_vas
        
    def evaluate_sa(self):
        sa_score, auc_all_sa = self.benchmark_sa.evaluate_all()
        self.logger.log_metrics({'SA':sa_score})
        df = pd.DataFrame(auc_all_sa)
        df.rename(columns={'auc':'global_auc'},inplace=True)
        df_exploded = df.explode('individual_auc').reset_index(drop=True)
        normalized_data = pd.json_normalize(df_exploded['individual_auc'])
        df_result = pd.concat([df_exploded, normalized_data], axis=1)
        df_result.drop('individual_auc', axis=1, inplace=True)
        df_result = df_result[df_result['stiffness_reduction']==0.03][['system_name','auc']]
        df_result.set_index('system_name',inplace=True)
        dict_auc_003= df_result.to_dict()['auc']
        dict_auc_003 = {f"SA_System_0.03: {key}":value for key,value in dict_auc_003.items()}
        self.logger.log_metrics(dict_auc_003)
    

        df_plot = self.benchmark_sa.combined_df
        df_grouped = df_plot.groupby(self.hue)
        for name, group in df_grouped:
            title = f"System {name}"
            fig,ax=plot_boxplot(group,axis_label=['stiffness_reduction','anomaly_index'],title=title)
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            self.logger.experiment.log_image(buf, name=title)
            plt.close(fig)
            buf.close()
        return sa_score
    


