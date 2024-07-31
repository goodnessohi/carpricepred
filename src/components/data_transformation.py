import sys
import os
from dataclasses import dataclass
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
    # Path to save the preprocessor object
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating the data transformation pipeline
        '''
        try:
            # Define numerical and categorical columns
            numerical_columns = ['year', 'km_driven', 'mileage', 'engine', 'seats', 'max_power']
            categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner',  'brand']
            logging.info('Loaded the categorical and numerical columns into the respective variables')
            
            # Pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),  # Use mean for imputation
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logging.info(f'Numerical columns scaling completed: {numerical_columns}')
            
            # Pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ]
            )
            logging.info(f'Categorical columns encoding completed: {categorical_columns}')

            # Combine pipelines into a preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def clean_column(self, df, column_name):
        """
        Cleans and converts the specified column to float.
        """
        # Define a custom function to extract numeric values
        def extract_numeric(value):
            if isinstance(value, str):
                # Remove any non-numeric characters, retain only the number
                cleaned_value = ''.join(c for c in value if c.isdigit() or c == '.')
                return float(cleaned_value) if cleaned_value else np.nan
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                return np.nan

        # Apply the function to the column
        df[column_name] = df[column_name].apply(extract_numeric)

        # Log after cleaning
        logging.info(f"Cleaned non-numeric values in column: {column_name}")
        return df

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f'Read train and test data completed')

            # Log the columns
            logging.info(f"Train data columns: {list(train_df.columns)}")
            logging.info(f"Test data columns: {list(test_df.columns)}")

            # Replace empty strings with NaN
            train_df.replace('', np.nan, inplace=True)
            test_df.replace('', np.nan, inplace=True)
            logging.info('Replaced empty strings with NaN')

            # Log the number of NaNs in each column
            logging.info(f"NaN values in train data: {train_df.isna().sum()}")
            logging.info(f"NaN values in test data: {test_df.isna().sum()}")

            # Clean specific columns with potential issues
            train_df = self.clean_column(train_df, 'max_power')
            test_df = self.clean_column(test_df, 'max_power')

            # Replace NaN values with a suitable replacement value (e.g., mean)
            train_df['max_power'] = train_df['max_power'].fillna(train_df['max_power'].mean())
            test_df['max_power'] = test_df['max_power'].fillna(test_df['max_power'].mean())

            # Ensure numerical columns are of numeric types
            for column in ['year', 'km_driven', 'mileage', 'engine', 'seats', 'max_power']:
                train_df[column] = pd.to_numeric(train_df[column], errors='coerce')
                test_df[column] = pd.to_numeric(test_df[column], errors='coerce')

            # Convert all categorical columns to strings
            categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
            for column in categorical_columns:
                train_df[column] = train_df[column].astype(str)
                test_df[column] = test_df[column].astype(str)

            # Separate input features and target features
            target_column_name = 'selling_price'
            if target_column_name not in train_df.columns:
                raise CustomException(f"Target column '{target_column_name}' not found in train data", sys)
            if target_column_name not in test_df.columns:
                raise CustomException(f"Target column '{target_column_name}' not found in test data", sys)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[[target_column_name]]
            target_feature_test_df = test_df[[target_column_name]]

            # Obtain preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Apply preprocessing
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Verify if any data still contains NaNs or wrong types
            try:
                logging.info(f"NaN values in input_feature_train_arr: {np.isnan(input_feature_train_arr).any()}")
                logging.info(f"NaN values in target_feature_train_df values: {np.isnan(target_feature_train_df.values).any()}")
                logging.info(f"NaN values in input_feature_test_arr: {np.isnan(input_feature_test_arr).any()}")
                logging.info(f"NaN values in target_feature_test_df values: {np.isnan(target_feature_test_df.values).any()}")
            except TypeError as e:
                logging.error(f"TypeError during NaN check: {e}")

            # Concatenate train array with target column
            train_arr = np.concatenate((input_feature_train_arr, target_feature_train_df.values.reshape(-1, 1)), axis=1)

            # Concatenate test array with target column
            test_arr = np.concatenate((input_feature_test_arr, target_feature_test_df.values.reshape(-1, 1)), axis=1)

            # Save the preprocessor object
            logging.info(f"Saving preprocessing object...")
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Applied preprocessing object on training and testing datasets.")    
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error(f"An error occurred during data transformation: {str(e)}")
            raise CustomException(str(e), sys)

if __name__ == "__main__":
    train_data_path = "artifacts/train.csv"
    test_data_path = "artifacts/test.csv"
    data_transformation = DataTransformation()

    try:
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info(f"Data transformation completed. Preprocessor saved at {preprocessor_path}")
    except CustomException as ce:
        logging.error(f"Data transformation failed: {str(ce)}")
