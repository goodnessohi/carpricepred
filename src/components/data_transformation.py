import sys
import os
from dataclasses import dataclass
import dill
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from  sklearn.impute import SimpleImputer 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj 

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['year', 'km_driven', 'mileage', 'engine', 'seats']
            categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'max_power', 'brand']
            logging.info('loaded the categorical and numerical columns into the respective variables')
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=False))

                ]
            )
            logging.info(f'Numerical columns scaling completed, {numerical_columns}')
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))

                ]
            )
            logging.info(f'Categorical columns encoding completed:{categorical_columns}')


            preprocessor= ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline,numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f'Read train and test data completed')

            logging.info(f'Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name = 'selling_price'
            numerical_columns = ['year', 'km_driven', 'mileage', 'engine', 'seats']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            target_feature_train_df = train_df[target_column_name]
            logging.info(f'Applying preprocessing object on training dataframe and testing datatframe')

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saving preprocessing object...")

            save_obj(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Applied preprocessing object on training and testing datasets.")    
            logging.info(
                f"Shape of train array: {train_arr.shape}, shape of test array: {test_arr.shape}"
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(str(e))
        
if __name__ == '__main__':
    obj = DataTransformation()
    train_path = 'path_to_train_file.csv'  # replace with your train file path
    test_path = 'path_to_test_file.csv'  # replace with your test file path
    transformed_data = obj.initiate_data_transformation(train_path, test_path)
    print(transformed_data)