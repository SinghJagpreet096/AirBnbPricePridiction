import sys
import os
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join('artifacts','model.pkl')
        self.preprocessor_path = os.path.join('artifacts','preprocessor.pkl')

    def predict(self, features):
        try:
            preprocessor = load_object(file_path = self.preprocessor_path)
            model = load_object(file_path = self.model_path)

            data_scaled = preprocessor.transform(features)

            preds = model.predict(data_scaled)

            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 smartlocation: str,
                 roomtype: str,
                 minimumnights: int,
                 availability365: int,
                 numberofreviews: int,
                 reviewscoresrating:int,
                 cancellationpolicy:str):
        self.smartlocation = smartlocation,
        self.roomtype = roomtype
        self.minimumnights = minimumnights
        self.availability365 = availability365
        self.numberofreviews = numberofreviews
        self.reviewscoresrating = reviewscoresrating
        self.cancellationpolicy = cancellationpolicy
        
    def get_data_as_data_frame(self):
        try:
            custom_data_input = {
                "smartlocation":[self.smartlocation],
                "roomtype":[self.roomtype],
                "minimumnights":[self.minimumnights],
                "availability365":[self.availability365],
                "numberofreviews":[self.numberofreviews],
                "reviewscoresrating":[self.reviewscoresrating],
                "cancellationpolicy":[self.cancellationpolicy]
            }
            return pd.DataFrame(custom_data_input)
        except Exception as e:
            raise CustomException(e, sys)

# if __name__ == "__main__":
#     obj = PredictPipeline()
#     sample = pd.read_csv('artifacts/test.csv')
#     print(obj.predict(features=sample))


        