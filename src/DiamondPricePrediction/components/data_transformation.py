import pandas as pd
import numpy as np
import os
import sys


from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customException


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.DiamondPricePrediction.utils.utils import save_object


from dataclasses import dataclass

@dataclass
class DataTransformationConfiguration:
     preproccessor_obj_path = os.path.join("artifacts", 'preprocessor.pkl')


class DataTransformation:
     def __init__(self):
          self.transformation_config = DataTransformationConfiguration()
     
     def get_data_transformation(self):
          try:
               logging.info("Data transformation initiated")
               
               cat_cols = ['cut', 'color', 'clarity']
               num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
               
               logging.info("pipeline initiated")
               num_pipeline = Pipeline(
                    steps=[
                         ('impute', SimpleImputer(strategy='mean'))
                         ('scaler', StandardScaler())
                    ]
               )
               cat_pipeline = Pipeline(
                    steps=[
                         ('imputer', SimpleImputer(strategy='most_frequent')),
                         ('encoder', OrdinalEncoder())
                    ]
               )
               
               preproccessor = ColumnTransformer(
                    transformers=[
                         ('numerical_pipeline', num_pipeline, num_cols),
                         ('categorical_pipeline', cat_pipeline, cat_cols)
                    ]
               )
               return preproccessor
               
          except Exception as e:
               logging.info("Exception occured in get data transformation function of data transformation file")
               raise customException(e,sys)
          
     def initiate_data_transformation(self, train_path,test_path):
          try:
               train_df = pd.read_csv(train_path)
               test_df = pd.read_csv(test_path)
               logging.info("I have read the train and test data path")
               logging.info("Train dataframe head \n",train_df.head())
               logging.info("test dataframe head \n", test_df.head())
               
               
               
               preprocceser_obj = self.get_data_transformation()
               
               target_coloum = "price"
               drop_coloumns = [target_coloum, 'id']
               
               input_fetaure_train_df = train_df.drop(columns=drop_coloumns, axis=1)
               input_feature_test_df = test_df.drop(columns=drop_coloumns, axis=1)
               target_feature_train_df = train_df[target_coloum]
               target_feature_test_df = test_df[target_coloum]
               
               
               input_feature_train_arr = preprocceser_obj.fit_transform(input_fetaure_train_df)
               input_feature_test_arr = preprocceser_obj.transform(input_feature_test_df)
               
               logging.info("Applying preproccesor object on train and test dataset")
               
               train_arr = np._c[input_feature_train_arr, np.array(target_feature_train_df)]
               test_arr = np._c[input_feature_test_arr, np.array(), np.array(target_feature_test_df)]
               
               
               
               save_object(
                    file_path = self.transformation_config.preproccessor_obj_path,
                    obj = preprocceser_obj
               )
               
               
               logging.info('preproccesing pickle file saved')
               
               
               return (
                    train_arr,
                    test_arr
               )
               
          except Exception as e:
               logging.info("Exception occured in initiate data transformation of data transformation file")
               raise customException(e,sys)