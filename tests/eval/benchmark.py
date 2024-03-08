import pytest

from src.downstream.base_dds import BaseDDS
from src.eval.benchmark import Benchmark_VAS
from unittest.mock import Mock, patch

# Mocking DataLoader and LightningDataModule for the sake of simplicity
# Adjust according to your actual data loading mechanism
MockDataLoader = Mock()
MockDataModule = Mock()
MockDataModule.before_virtual_anomaly.return_value = MockDataLoader
MockDataModule.after_virtual_anomaly.return_value = MockDataLoader

# Example test to check if Benchmark_VAS initializes correctly
def test_benchmark_vas_initialization():
    dds = Mock(spec=BaseDDS)
    datamodule = MockDataModule

    benchmark_vas = Benchmark_VAS(dds, datamodule=datamodule)
    assert benchmark_vas.dds == dds
    assert benchmark_vas.before_virtual_anomaly == MockDataLoader
    assert benchmark_vas.after_virtual_anomaly == MockDataLoader

# Example test for setup_anomaly_score method
@patch('src.eval.benchmark_vas.get_anomaly_score')
def test_setup_anomaly_score(mock_get_anomaly_score):
    mock_get_anomaly_score.return_value = {'condition': ['before', 'after'], 'anomaly_index': [0.1, 0.2]}
    dds = Mock(spec=BaseDDS)
    datamodule = MockDataModule

    benchmark_vas = Benchmark_VAS(dds, datamodule=datamodule)
    benchmark_vas.setup_anomaly_score(['freq'], ['amplitude'])

    assert 'condition' in benchmark_vas.combined_df.columns
    assert 'anomaly_index' in benchmark_vas.combined_df.columns
    assert len(benchmark_vas.combined_df) == 2  # Assuming your mock data has two entries

# Add more tests to cover other methods like evaluate_individu and evaluate_all

if __name__ == "__main__":
    test_benchmark_vas_initialization()
    test_setup_anomaly_score()