import os
import sys
import dill
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from src.exception import CustomException

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise CustomException(str(e))
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    report = {}
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)  # Train model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        except Exception as e:
            print(f"Error evaluating model {model_name}: {str(e)}")
    return report

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)