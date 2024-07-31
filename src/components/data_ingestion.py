import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.exception import CustomException
from src.logger import logging
import pathlib

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            # Read the dataset
            logging.info('Attempting to read the dataset')
            df = pd.read_csv('notebook/data/vehicle_data_cleaned.csv')
            logging.info('Read the dataset into a dataframe')

            # Create the directory if it doesn't exist
            logging.info('Attempting to create directory for artifacts')
            pathlib.Path('artifacts').mkdir(parents=True, exist_ok=True)
            logging.info('Created directory for artifacts if not already present')

            # Save the raw data
            logging.info(f'Attempting to save raw data to {self.ingestion_config.raw_data_path}')
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f'Saved raw data to {self.ingestion_config.raw_data_path}')

            # Train test split
            logging.info('Attempting to split the data into train and test sets')
            train, test = train_test_split(df, test_size=0.2, random_state=42)
            logging.info('Data split into train and test sets')

            logging.info(f'Attempting to save train data to {self.ingestion_config.train_data_path}')
            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logging.info(f'Saved train data to {self.ingestion_config.train_data_path}')

            logging.info(f'Attempting to save test data to {self.ingestion_config.test_data_path}')
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f'Saved test data to {self.ingestion_config.test_data_path}')

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error('Error occurred in data ingestion config')
            raise CustomException(str(e), sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, *additional_values =  data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
