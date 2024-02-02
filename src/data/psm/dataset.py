from typing import Callable, List, Union, Optional
from pathlib import Path
import sqlite3
from torch.utils.data import Dataset
import torch
import numpy as np
from contextlib import closing
from omegaconf import DictConfig, OmegaConf
import hydra

class PSMDatasetBuilder:
    def __init__(self):
        self.database_path: Union[Path, str] = None
        self.table_name: str = 'processed_data'
        self.columns: List[str] = ['*']  # Default to all columns if not specified
        self.conditions: str = ''
        self.parameters: List = []
        self.transform: Optional[Callable] = None
        self.transform_label: Optional[Callable] = None
        self.preload: bool = False

    def set_database_path(self, path: Union[str, Path]) -> 'PSMDatasetBuilder':
        self.database_path = path
        return self

    def set_table_name(self, table_name: str) -> 'PSMDatasetBuilder':
        self.table_name = table_name
        return self

    def set_columns(self, columns: List[str]) -> 'PSMDatasetBuilder':
        self.columns = columns
        return self

    def add_condition(self, condition: str, params: List) -> 'PSMDatasetBuilder':
        if self.conditions:
            self.conditions += ' AND '
        self.conditions += condition
        self.parameters += params
        return self

    def set_transform(self, transform: Callable) -> 'PSMDatasetBuilder':
        self.transform = transform
        return self

    def set_transform_label(self, transform_label: Callable) -> 'PSMDatasetBuilder':
        self.transform_label = transform_label
        return self

    def enable_preloading(self, preload: bool = True) -> 'PSMDatasetBuilder':
        self.preload = preload
        return self
    # let's make the query to the database that finds the keys that match the condition
    # then we can use the keys to fetch the data

    
    def build(self) -> Dataset:
 
        return PSMDataset(self)

class PSMDataset(Dataset):
    def __init__(self, builder: PSMDatasetBuilder):
        self.database_path = builder.database_path
        self.table_name = builder.table_name
        self.columns = ', '.join(builder.columns)
        self.condition = builder.conditions
        self.parameters = builder.parameters
        self.transform = builder.transform
        self.transform_label = builder.transform_label
        self.preload = builder.preload


        self.keys = self._get_keys()

        self.data = self._preload_data() if self.preload else None
    def _get_keys(self):
        query = f"SELECT id FROM {self.table_name} WHERE {self.condition}" if self.condition else f"SELECT id FROM {self.table_name}"
        with closing(sqlite3.connect(self.database_path)) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute(query, self.parameters)
                return [row[0] for row in cursor.fetchall()]
            
    def _preload_data(self):
        data = []
        conn= sqlite3.connect(self.database_path)
        cursor= conn.cursor()
                # Building a query to fetch all rows at once based on the keys
        placeholders = ','.join('?' for _ in self.keys)
        query = f"SELECT {self.columns} FROM {self.table_name} WHERE id IN ({placeholders})"
        cursor.execute(query, self.keys)
        rows = cursor.fetchall()
        for row in rows:
            data.append(self._process_row(row))
        # print size of data in MB
        return data

    def __len__(self):
        return len(self.keys)
    def __getitem__(self, idx):
        if self.preload:
            return self.data[idx]
        else:
            key = self.keys[idx]
            row = self._fetch_data(key)
            return self._process_row(row) if row else None
    def _fetch_data(self, key):
        query = f"SELECT {self.columns} FROM {self.table_name} WHERE id = ?"
        with closing(sqlite3.connect(self.database_path)) as conn:
            with closing(conn.cursor()) as cursor:
                cursor.execute(query, (key,))
                return cursor.fetchone()
    def _process_row(self, row):
        res = []
        for data, tsf in zip(row, self.transform):
            if tsf:
                res.append(tsf(data))
            else:
                res.append(data)

        return tuple(res)
    
def build_dataset(database_path: Union[str, Path],
                  table_name: str = 'processed_data',
                  columns: List[str] = ['PSD', 'system_name'],
                  condition: str = '',
                  parameters: List[str] = [],
                  transform: List[Callable] = None,
                  preload: bool = False) -> Dataset:

    assert len(columns)== len(transform), "The number of columns and transforms must be the same"

    dataset =  PSMDatasetBuilder()\
        .set_database_path(database_path)\
        .set_table_name(table_name)\
        .set_columns(columns)\
        .add_condition(condition, parameters)\
        .set_transform(transform)\
        .enable_preloading(preload)\
        .build()
    return dataset
