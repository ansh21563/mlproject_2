import dagshub
dagshub.init(repo_owner="ansh21563", repo_name="mlproject_2", mlflow=True)

import mlflow

with mlflow.start_run():
    mlflow.log_param("test_param", "hello")
    mlflow.log_metric("test_metric", 123)
