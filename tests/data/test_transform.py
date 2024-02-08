import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
from src.data.psm.transform import CreateTransformer  # Adjust import path as needed

class TestCreateTransformer(unittest.TestCase):
    def setUp(self):
        self.database_path = "dummy_db_path.db"
        self.freq_min = 0
        self.freq_max = 4
        self.num_classes = 200
        self.freq_data = np.linspace(1, 10, 10, dtype=np.float32)
        self.psd_data = np.random.rand(10).astype(np.float32)

        # Mock return values for sqlite3 cursor fetchone and fetchall
        self.fetchone_return = (self.freq_data.tobytes(),)
        self.fetchall_return = [(self.psd_data.tobytes(),) for _ in range(5)]

    @patch('sqlite3.connect')
    def test_initialization_and_params(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = self.fetchone_return
        mock_cursor.fetchall.return_value = self.fetchall_return

        transformer = CreateTransformer(self.database_path, self.freq_min, self.freq_max, self.num_classes)

        # Assert _initialize_params works as expected
        self.assertIsNotNone(transformer.freq_before)
        self.assertTrue((transformer.freq_before >= self.freq_min).all() and (transformer.freq_before <= self.freq_max).all())
        self.assertGreaterEqual(transformer.min_psd, 0)
        self.assertLessEqual(transformer.max_psd, np.log(self.psd_data).max())
        self.assertEqual(transformer.dim_psd, np.log(self.psd_data[self.psd_data >= self.freq_min]).shape[0])

    @patch('sqlite3.connect')
    def test_get_transformer_psd(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = self.fetchone_return
        mock_cursor.fetchall.return_value = self.fetchall_return

        transformer = CreateTransformer(self.database_path, self.freq_min, self.freq_max, self.num_classes)
        transform_psd = transformer.get_transformer_psd()

        # Simulate PSD data transformation
        sample_psd = np.random.rand(10).astype(np.float32).tobytes()
        transformed_psd = transform_psd(sample_psd)

        self.assertIsInstance(transformed_psd, torch.FloatTensor)

    def test_get_transformer_label(self):
        transformer = CreateTransformer(self.database_path, self.freq_min, self.freq_max, self.num_classes)
        transform_label = transformer.get_transformer_label()

        # Test label extraction
        sample_label = "class_label_2"
        extracted_label = transform_label(sample_label)

        self.assertEqual(extracted_label, 2)

if __name__ == '__main__':
    unittest.main()
