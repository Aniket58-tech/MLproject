import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import (
    save_object,
    evaluate_models
)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input and target variables")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1], 
                test_array[:, -1]
            )
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }

            params={
                "Decision Tree":{
                    'criterion':["squared_error",'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best', 'random'],
                    # 'max_features':['auto', 'sqrt', 'log2'],
                },
                "Random Forest":{
                    #'criterion':["squared_error",'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['auto', 'sqrt', 'log2'],
                    'n_estimators':[8,16,32,64,128,256],
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'learning_rate':[0.01, 0.1, 0.2],
                    'subsample':[0.6, 0.8, 1.0],
                    #'criterion':['friedman_mse', 'squared_error'],
                    # 'max_features':['auto', 'sqrt', 'log2'],
                    'n_estimators':[8,16,32,64,128,256],
                },
                "Linear Regression":{},
                "KNN":{
                    'n_neighbors':[3,5,7,9],
                    #'weights':['uniform', 'distance'],
                    #'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "AdaBoost":{
                    'learning_rate':[0.01, 0.1, 0.2],
                    # 'loss':['linear', 'square', 'exponential'],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "XGBoost":{
                    'learning_rate':[0.01, 0.1, 0.2],
                    'n_estimators':[8,16,32,64,128,256],
                    #'max_depth':[3, 5, 7],
                    #'subsample':[0.6, 0.8, 1.0]
                },
                "CatBoost":{
                    'learning_rate':[0.01, 0.1, 0.2],
                    'depth':[3, 5, 7],
                    'iterations':[8,16,32,64,128,256],
                    #'l2_leaf_reg':[1, 3, 5]
                }
            }

            model_report:dict=evaluate_models(X_train, y_train, X_test, y_test, models, param=params)
            
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with sufficient accuracy", sys)
            
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            preprocessing_obj = save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
            