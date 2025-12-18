from src.MLPROJECT_2.logger import logging
from src.MLPROJECT_2.exception import CustomException

from src.MLPROJECT_2.components.data_ingestion import DataIngestion
from src.MLPROJECT_2.components.data_ingestion import DataIngestionConfig
from src.MLPROJECT_2.components.data_tansformation import DataTransformation,DataTransformationConfig
from src.MLPROJECT_2.components.model_trainer import ModelTrainerConfig,ModelTainer 
import sys

if __name__=="__main__":
    logging.info("The execution has started")


    try:
        #data_ingestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        
        # data_tranformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        
        ## Model Training
        model_trainer=ModelTainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        
    except Exception as e:
      logging.info("Custom Exception")
      raise CustomException(e,sys)