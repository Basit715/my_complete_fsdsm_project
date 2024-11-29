from src.DiamondPricePrediction.pipelines.prediction_pipeline import customData


custom_data_object = customData(1.52,62.2,58.0,7.27,7.33,4.55,'Premium','F','VS2')


data = custom_data_object.get_data_as_dataframe()


print(data)
