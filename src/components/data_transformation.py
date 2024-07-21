import sys
import os
from dataclasses import dataclass
import dill
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['year', 'km_driven', 'mileage', 'engine', 'seats']
            categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'max_power', 'brand']
            logging.info('Loaded the categorical and numerical columns into the respective variables')
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f'Numerical columns scaling completed: {numerical_columns}')
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f'Categorical columns encoding completed: {categorical_columns}')

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f'Read train and test data completed')
            

            logging.info(f'Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'selling_price'
            numerical_columns = ['year', 'km_driven', 'mileage', 'engine', 'seats']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]
            logging.info(f'Applying preprocessing object on training dataframe and testing dataframe')
            logging.info(f"Shape of input_feature_train_df after separation: {input_feature_train_df.shape}")
            logging.info(f"Shape of target_feature_train_df after separation: {target_feature_train_df.shape}")
            logging.info(f"Shape of input_feature_test_df after separation: {input_feature_test_df.shape}")
            logging.info(f"Shape of target_feature_test_df after separation: {target_feature_test_df.shape}")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            logging.info(f"Shape of input_feature_train_arr after transform: {input_feature_train_arr.shape}")
            logging.info(f"Shape of input_feature_test_arr after transform: {input_feature_test_arr.shape}")

            logging.info(f"Saving preprocessing object...")
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Applied preprocessing object on training and testing datasets.")    

            return (
                input_feature_train_arr,
                target_feature_train_df.values,
                input_feature_test_arr,
                target_feature_test_df.values,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error(f"An error occurred during data transformation: {str(e)}")
            raise CustomException(str(e), sys)

# Main execution
if __name__ == "__main__":
    try:
        data_transformation = DataTransformation()
        train_data_path = 'path_to_train_data.csv'  # Update with actual path
        test_data_path = 'path_to_test_data.csv'    # Update with actual path
        input_feature_train_arr, target_feature_train_arr, input_feature_test_arr, target_feature_test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
