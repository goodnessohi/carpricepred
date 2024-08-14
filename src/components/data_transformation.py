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
            categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']
            logging.info('Loaded the categorical and numerical columns into the respective variables')
            
            # Pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Use most frequent for imputation
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
        def extract_numeric(value):
            if isinstance(value, str):
                cleaned_value = ''.join(c for c in value if c.isdigit() or c == '.')
                return float(cleaned_value) if cleaned_value else np.nan
            elif isinstance(value, (int, float)):
                return float(value)
            else:
                return np.nan

        df[column_name] = df[column_name].apply(extract_numeric)
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

            X_train = train_df.drop(columns=[target_column_name], axis=1)
            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[[target_column_name]]
            y_test = test_df[[target_column_name]]

            # Obtain preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Apply preprocessing
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            # Verify if any data still contains NaNs or wrong types
            try:
                logging.info(f"Data type of X_train_transformed: {X_train_transformed.dtype}")
                logging.info(f"Data type of y_train: {y_train.dtypes}")
                logging.info(f"Data type of X_test_transformed: {X_test_transformed.dtype}")
                logging.info(f"Data type of y_test: {y_test.dtypes}")
                
                if np.issubdtype(X_train_transformed.dtype, np.number):
                    logging.info(f"NaN values in X_train_transformed: {np.isnan(X_train_transformed).any()}")
                
                if np.issubdtype(y_train.values.dtype, np.number):
                    logging.info(f"NaN values in y_train: {np.isnan(y_train.values).any()}")
                
                if np.issubdtype(X_test_transformed.dtype, np.number):
                    logging.info(f"NaN values in X_test_transformed: {np.isnan(X_test_transformed).any()}")
                
                if np.issubdtype(y_test.values.dtype, np.number):
                    logging.info(f"NaN values in y_test: {np.isnan(y_test.values).any()}")
            except TypeError as e:
                logging.error(f"TypeError during NaN check: {e}")

            # Save the preprocessor object
            logging.info(f"Saving preprocessing object...")
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info(f"Applied preprocessing object on training and testing datasets.")    
            
            return (
                X_train_transformed,
                X_test_transformed,
                y_train.values,
                
                y_test.values,
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
        X_train, y_train, X_test, y_test, preprocessor_path = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info(f"Data transformation completed. Preprocessor saved at {preprocessor_path}")
    except CustomException as ce:
        logging.error(f"Data transformation failed: {str(ce)}")
