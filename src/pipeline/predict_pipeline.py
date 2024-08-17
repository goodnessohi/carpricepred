import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging
import os

class PredictPipeline:
    def __init__(self):
        
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            logging.info('Model and preprocessor path obtained')
            
            model = load_object(file_path=model_path)    
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)
        

class CustomData:
    def __init__(self,
                year: int,
                km_driven: int,
                mileage: float,
                engine: float,
                seats: int,
                max_power: float,
                fuel: str,
                seller_type: str,
                transmission: str,
                owner: str,
                brand: str
                 ):
        self.year = year
        self.km_driven = km_driven
        self.mileage = mileage
        self.engine = engine
        self.seats = seats
        self.max_power = max_power
        self.fuel = fuel
        self.seller_type = seller_type
        self.transmission = transmission
        self.owner = owner
        self.brand = brand

    def get_data_as_df(self):
        try:
            custom_data_input_dict = {
            "year": [self.year],
            "km_driven": [self.km_driven],
            "mileage": [self.mileage],
            "engine": [self.engine],
            "seats": [self.seats],
            "max_power": [self.max_power],
            "fuel": [self.fuel],
            "seller_type": [self.seller_type],
            "transmission": [self.transmission],
            "owner": [self.owner],
            "brand": [self.brand],
        }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
                print(f"Original error: {str(e)}")  # Log the original error
                raise CustomException(e, sys)

