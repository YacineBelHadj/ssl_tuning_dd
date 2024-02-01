import pytest
from unittest.mock import MagicMock, patch
from src.data.psm.dataset import PSMDatasetBuilder, build_dataset  # Adjust import path as necessary

# Mocked row data to simulate database fetch operations
mocked_rows = [
    (b'\x00\x00\x00\x00', 'system1', 0),  # Mocked PSD data, system name, and anomaly level
    (b'\x01\x01\x01\x01', 'system2', 1),
]

@pytest.fixture
def mock_sqlite(mocker):
    """
    Fixture to mock sqlite3 connection and cursor objects.
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchall.return_value = mocked_rows
    mock_cursor.fetchone.return_value = mocked_rows[0]
    
    with patch('sqlite3.connect', return_value=mock_conn):
        yield mock_cursor

def test_dataset_initialization(mock_sqlite):
    """
    Test that the dataset initializes correctly and can fetch data.
    """
    dataset = build_dataset(
        database_path=':memory:',  # Using in-memory database for simplicity
        table_name='test_table',
        columns=['PSD', 'system_name', 'anomaly_level'],
        preload=True
    )

    # Test __len__ method
    assert len(dataset) == len(mocked_rows), "Dataset length should match the number of mocked rows."

    # Test __getitem__ method for the first item
    first_item = dataset[0]
    assert first_item, "Dataset should be able to fetch an item."

    # Assuming the transform functions are simple pass-through (or mock them if necessary)
    assert first_item[0].shape[0] == 4, "The PSD data should be correctly transformed into a tensor."

def test_lazy_loading(mock_sqlite):
    """
    Test that lazy loading fetches data correctly when accessed.
    """
    dataset = build_dataset(
        database_path=':memory:',
        table_name='test_table',
        columns=['PSD', 'system_name', 'anomaly_level'],
        preload=False
    )

    # Access an item to trigger lazy loading
    item = dataset[0]
    assert item, "Dataset should lazily load an item."

    # Further checks can be added to validate data integrity and transformations

