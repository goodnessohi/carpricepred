import os
import sys
from dataclasses import dataclass
#Modelling
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
#Metrics
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info('Splitting train and test data')
            X_train, y_train, X_test, y_test = (
            train_arr[:, :-1],
            train_arr[:, -1],
            test_arr[:, :-1],
            test_arr[:, -1]
            )

            models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(), 
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor()
        }
            
            params = {
                "Linear Regression": {},
                "Lasso": {"alpha": [0.1, 1.0, 10.0]},
                "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                "K-Neighbors Regressor": {"n_neighbors": [3, 5, 7, 9]},
                "Decision Tree": {"max_depth": [None, 10, 20, 30]},
                "Random Forest Regressor": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
                "XGBRegressor": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1, 0.3]},
                "CatBoosting Regressor": {"iterations": [100, 200], "depth": [6, 8, 10]},
                "AdaBoost Regressor": {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 1.0]}
            }
            best_models = {}
            best_params = {}
            for model_name in models:
                logging.info(f'Tuning hyperparameters for {model_name}')
                grid_search = GridSearchCV(models[model_name], params[model_name], cv=3, n_jobs=-1, scoring='r2')
                grid_search.fit(X_train, y_train)
                best_models[model_name] = grid_search.best_estimator_
                best_params[model_name] = grid_search.best_params_
                logging.info(f'Best hyperparameters for {model_name}: {grid_search.best_params_}')

            
            model_report = evaluate_models(X_train, y_train, X_test,y_test, best_models)
            logging.info('Model evaluation is completed.')

            #Get model score
            best_model_score = max(model_report.values())
            logging.info('Best model score has been obtained')

            #Get best model from dictionary
            best_model_name = max (model_report, key= model_report.get)
            best_model = best_models[best_model_name]
            best_hyperparameters = best_params[best_model_name]

            if best_model_score < 0.6:
                raise CustomException(f"No best model as the model scores are less than 60%")
            logging.info(f'Best model found on training and test datasets')

            #Print best hyperparams
            logging.info(f'Best hyperparamters for the best model {best_model_name} and the hyperparameter is : {best_hyperparameters}')

            save_obj(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info('Trained model is used to make predictions on test data.')
            return r2_square
        except Exception as e:
            raise CustomException(str(e))
    