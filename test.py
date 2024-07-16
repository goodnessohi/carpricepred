import unittest
from src.components.data_ingestion import DataIngestion

class TestDataIngestion(unittest.TestCase):
    def test_instantiation(self):
        data_ingestion = DataIngestion()
        self.assertIsInstance(data_ingestion, DataIngestion)

    def test_ingest_data(self):
        data_ingestion = DataIngestion()
        data_ingestion.ingest_data()  # Assume this method runs without errors
        self.assertTrue(True)  # If no errors, the test passes

    def test_process_data(self):
        data_ingestion = DataIngestion()
        data_ingestion.process_data()  # Assume this method runs without errors
        self.assertTrue(True)  # If no errors, the test passes

if __name__ == '__main__':
    unittest.main()