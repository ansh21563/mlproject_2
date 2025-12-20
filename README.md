## END-TO-END DATA SCIENCE PROJECT
import dagshub
dagshub.init(repo_owner='ansh21563', repo_name='mlproject_2', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)