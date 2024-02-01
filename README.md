<div align="center">

# Welcome! 👋 
### Implementing Self-Supervised Learning (SSL) for Damage Detection

</div>

## Introduction
This project leverages `pytorch-lightining` and `hydra` to create a framework for training and tuning models specifically for damage detection tasks. It is designed to be flexible and easily adaptable to different damage detection datasets.

## Installation
To get started with this project, clone the repository and install the required dependencies:
```bash
git clone [ssl][https://github.com/YacineBelHadj/SSL_DD]
cd [ssl][SSL_DD]
pip install -r requirements.txt
```

## Configuration
All configuration are managed through a config file. Ensure you specify your data paths and model settings in `configs/` folder using the hydra config format. For example, to specify the data path, create a file `conf/data.yaml` with the following content:
```yaml
data:
  path: /path/to/data
model:
  name: resnet18
  grid_search:
    - name: lr
      values: [0.001, 0.01, 0.1]
    - name: batch_size
      values: [32, 64, 128]
    - depth:
      values: [18, 34, 50]
    - layers_size:
      values: [64, 128, 256]
    - dropout:
      values: [0.0, 0.1, 0.2]
    - batch_norm:
      values: [True, False]

processing: 
    transform_input: function_name
    transform_output: function_name
model path: /path/to/model

```
## Data Loading 

Your data needs to be specified in the config file. Additionally, you will need to create a custom data loading module. This module should inherit the `DataModule` class from `pytorch-lightning` and implement the follwing methods:
```python
class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def reference_dataloader(self):
        pass

    def affected_virtual_anomaly_dataloader(self):
        pass

    def affected_real_anomaly_dataloader(self):
        pass
```
## Usuage
To train a model, run the following command:
```bash
train:SSL_DD
test:SSL_DD
tune:SSL_DD
```
## Structure of the Project
The project is structured as follows:
```bash

├── configs
│   ├── data.yaml
│   ├── model.yaml
│   └── trainer.yaml
├── Logs
│   ├── default
├── ssl_dd
│   ├── data
│   │   ├── __init__.py
│   ├── models
│   │   ├── __init__.py
│   ├── processing
│   │   ├── __init__.py
│   ├── training 
│   │   ├── __init__.py
│   ├── tuning
│   │   ├── __init__.py
│   ├── utils
│   │   ├── __init__.py
│   ├── __init__.py
│   ├── data_module.py
│   ├── main.py
│   ├── model.py
│   ├── trainer.py
│   └── tuning.py
├── README.md
├── requirements.txt
└── setup.py
```

## Contributing

Contributions are always welcome! Please contact me if you are interested in contributing to this project. I am quite a noob and need all the help I can get! 

## License

[MIT](https://choosealicense.com/licenses/mit/)