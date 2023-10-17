import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging

## metrics
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]

            logging.info('training started')

            model.fit(X_train,y_train)

            ## training prediction
            y_train_pred = model.predict(X_train)

            ## testing prediction

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_true=y_train,y_pred=y_train_pred)

            test_model_score = r2_score(y_true=y_test, y_pred=y_test_pred)


            report[list(models.keys())[i]] = test_model_score
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)