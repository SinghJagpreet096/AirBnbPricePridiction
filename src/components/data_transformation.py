import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utils import save_object

## transformation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()
        self.categorical_columns = ['roomtype','cancellationpolicy','smartlocation']
        self.numerical_columns_ss = ['minimumnights','numberofreviews']
        self.numerical_columns_mm = ['availability365','reviewscoresrating']

        self.target_column_name = 'price'

    def get_data_transformer_object(self):
        """
        
        """

        try:
            numerical_transformer_ss = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='mean')),
                ('scaller',StandardScaler())
            ]) 
            
            numerical_transformer_mm = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('minmax',MinMaxScaler())
            ])

            categorical_tranformer = Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehot',OneHotEncoder(drop='first',handle_unknown='ignore'))
            ])

            logging.info(f"categorical columns: {self.categorical_columns}")
            logging.info(f"numerical columns: {self.numerical_columns_mm + self.numerical_columns_ss}")

            preprocessor = ColumnTransformer([
                ('numerical_trans_ss',numerical_transformer_ss,self.numerical_columns_ss),
                ('numerical_trans_mm',numerical_transformer_mm,self.numerical_columns_mm),
                ('categorical_trans',categorical_tranformer,self.categorical_columns)
            ])


            return preprocessor



        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_date_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('read train and test data')

            logging.info("obtaining preprocessor obj")

            preprocessor_obj  = self.get_data_transformer_object()

            input_feature_train_df = train_df.drop(self.target_column_name,axis=1)
            target_feature_train_df = train_df[self.target_column_name]

            input_feature_test_df = test_df.drop(self.target_column_name,axis=1)
            target_feature_test_df = test_df[self.target_column_name]

            logging.info("applying transformations")

            input_feature_train_array = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor_obj.transform(input_feature_test_df)

            logging.info(f'shape of train: {input_feature_train_array.shape} shape of y train: {target_feature_train_df.shape}')
            
            
            ## combine feature arr and target
            # train_arr = np.c_[
            #     np.array(input_feature_train_array), (target_feature_train_df)
            # ]

            # test_arr = np.c_[
            #     input_feature_test_array, (target_feature_test_df)
            # ]


            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessor_obj
            )

            logging.info(f"saved preprocessor")

            return (
                input_feature_train_array,
                target_feature_train_df,
                input_feature_test_array,
                target_feature_test_df,
                self.data_transformation_config.preprocessor_ob_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)