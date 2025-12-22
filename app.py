import sys
from src.MLPROJECT_2.logger import logging
from src.MLPROJECT_2.exception import CustomException
from src.MLPROJECT_2.components.data_ingestion import DataIngestion
from src.MLPROJECT_2.components.data_tansformation import DataTransformation
from src.MLPROJECT_2.components.model_trainer import ModelTrainer
import dagshub
import mlflow

if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        # ----------------------------
        # Data Ingestion
        # ----------------------------
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # ----------------------------
        # Data Transformation
        # ----------------------------
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        # ----------------------------
        # DagsHub + MLflow setup
        # ----------------------------
        # dagshub.init(
        #     repo_owner="ansh21563",
        #     repo_name="mlproject_2",
        #     mlflow=True
        # )

        # ----------------------------
        # Model Training
        # ----------------------------
        model_trainer = ModelTrainer()
        r2_score_value = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(r2_score_value)

    except Exception as e:
        logging.info("Custom Exception occurred")
        raise CustomException(e, sys)