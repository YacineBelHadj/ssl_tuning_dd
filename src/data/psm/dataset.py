from typing import Callable, List, Union, Optional
from pathlib import Path
import sqlite3
from torch.utils.data import Dataset
import torch
import numpy as np
from contextlib import closing


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
        self.conditions = condition
        self.parameters = params
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

    def build(self) -> Dataset:
        return PSMDataset(self)

class PSMDataset(Dataset):
    def __init__(self, builder: PSMDatasetBuilder):
        self.database_path = builder.database_path
        self.table_name = builder.table_name
        self.columns = ', '.join(builder.columns)
        self.conditions = builder.conditions
        self.parameters = builder.parameters
        self.transform = builder.transform
        self.transform_label = builder.transform_label
        self.preload = builder.preload
        self.data = []
        self.keys = []

        # Initialize dataset in a context manager
        with closing(sqlite3.connect(self.database_path)) as conn:
            with closing(conn.cursor()) as cursor:
                self._initialize_dataset(cursor)

    def _initialize_dataset(self, cursor):
        query_keys = f"SELECT id FROM {self.table_name}"
        condition_query = f" WHERE {self.conditions}" if self.conditions else ""

        # Preload keys
        cursor.execute(query_keys + condition_query, self.parameters)
        self.keys = [row[0] for row in cursor.fetchall()]

        if self.preload:
            base_query = f"SELECT {self.columns} FROM {self.table_name}{condition_query}"
            cursor.execute(base_query, self.parameters)
            self.data = cursor.fetchall()
            print('Preloading data...')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.preload:
            row = self.data[idx]
        else:
            with closing(sqlite3.connect(self.database_path)) as conn:
                with closing(conn.cursor()) as cursor:
                    key_query = f"SELECT {self.columns} FROM {self.table_name} WHERE id = ?"
                    cursor.execute(key_query, (self.keys[idx],))
                    row = cursor.fetchone()

        return self._process_row(row)

    def _process_row(self, row):
        result = []
        for i, col in enumerate(self.columns.split(', ')):
            data = row[i]
            if col == 'PSD' and data is not None:
                psd = np.frombuffer(data)
                if self.transform:
                    psd = self.transform(psd)
                psd = torch.from_numpy(psd).float()
                result.append(psd)
            elif col in ['system_name'] and self.transform_label:
                result.append(self.transform_label(data))
            else:
                result.append(data)
        return tuple(result)

def build_dataset(database_path: Union[str, Path],
                  table_name: str = 'processed_data',
                  columns: List[str] = ['PSD', 'system_name', 'anomaly_level'],
                  condition: str = '',
                  parameters: List = [],
                  transform: Callable = None,
                  transform_label: Callable = None,
                  preload: bool = False) -> Dataset:
    
    return PSMDatasetBuilder()\
        .set_database_path(database_path)\
        .set_table_name(table_name)\
        .set_columns(columns)\
        .add_condition(condition, parameters)\
        .set_transform(transform)\
        .set_transform_label(transform_label)\
        .enable_preloading(preload)\
        .build()
