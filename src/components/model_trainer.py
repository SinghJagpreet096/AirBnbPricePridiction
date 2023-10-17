import os
import sys  
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_models

from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.data_ingestion import DataIngestion,DataIngestionConfig


from dataclasses import dataclass

# models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso,Ridge, LinearRegression
from xgboost import XGBRegressor
## metrics
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr,target_train, test_arr,target_test):
        try:
            logging.info("spliting train and test input and target")

            X_train, y_train, X_test, y_test = (
                train_arr,
                target_train,
                test_arr,
                target_test
            )

            models = {
                "LinearRegression":LinearRegression(),
                # "Ridge":Ridge(),
                # "Lasso":Lasso(),
                # "DecisionTreeRegressor":DecisionTreeRegressor(),
                # "RandomForestRegressor":RandomForestRegressor(),
                # "AdaBoostRegressor":AdaBoostRegressor(),
                # "XGBRegressor":XGBRegressor()
            }

            ## TODO Hypertuning
            ## hyper parameters

            logging.info('training models')

            model_report:dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
                )
            
            logging.info('model trained')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            # if best_model_score < 0.7:
            #     raise CustomException("No best Model found",sys)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2 = r2_score(y_true=y_test, y_pred=predicted)

            return r2
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    data_ingestion = DataIngestion()

    train_data_path, test_data_path = data_ingestion.iniate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr,target_train, test_arr,target_test,_ = data_transformation.initiate_date_transformation(train_data_path,test_data_path)

    model_trainer = ModelTrainer()

    print(model_trainer.initiate_model_trainer(train_arr,target_train,test_arr,target_test))