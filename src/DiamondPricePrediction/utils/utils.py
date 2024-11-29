import pandas as pd
import numpy as np
import os
import sys
import pickle
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import customException

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


def save_object(file_path,obj):
     try:
          dir_path = os.path.dirname(file_path)
          os.makedirs(dir_path, exist_ok=True)
          with open(file_path, "wb") as file_obj:
               pickle.dump(obj, file_obj)
     except Exception as e:
          logging.info("Exception occured in pytho script utils.py")
          raise customException(e,sys)
     
     
def evaluate_model(x_train,y_train,x_test,y_test,models):
          try:
               report = {}
               for i in range(len(models)):
                    model = list(models.values())[i]
                    model.fit(x_train,y_train)
                    
                    
                    y_pred = model.predict(x_test)
                    test_model_score = r2_score(y_test,y_pred)
                    
                    
                    report[list(models.keys())[i]] = test_model_score
                    
               return report

                    
          except Exception as e:
               raise customException(e,sys)
          
def load_object(file_path):
     try:
          with open(file_path,'rb') as file_obj:
               return pickle.load(file_obj)
          
     except Exception as e:
          logging.info('Error occured in load_object of utils function')