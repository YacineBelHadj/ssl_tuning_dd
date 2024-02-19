
from src.downstream.base_dds import BaseDDS
from pytorch_lightning import LightningDataModule
from typing import List
from src.eval.benchmark_vas import Benchmark_VAS

class Benchmark:
    def __init__(self,dds:BaseDDS,
        datamodule:LightningDataModule=None,
        before_anomaly_columns:List[str]=None,
        after_anomaly_columns:List[str]=None,
        transform:str=None,
        hue:str=None,
        anomaly_config:List[str]=None):
        self.dds = dds
        self.datamodule = datamodule

        self.before_anomaly_columns = before_anomaly_columns
        self.after_anomaly_columns = after_anomaly_columns        
        self.transform = transform
        self.hue = hue
        self.anomaly_config = anomaly_config
        

    def setup(self):
        self.benchmark_vas = Benchmark_VAS(self.dds,
                                          datamodule=self.datamodule)
        
        self.benchmark_vas.setup_anomaly_score(self.before_anomaly_columns,
                                               self.after_anomaly_columns)
        
    def evaluate(self):
        df_res, vas = self.benchmark_vas.evaluate_all(hue=self.hue,transform=self.transform,
                                        anomaly_config=self.anomaly_config)
        return None,vas
        
        