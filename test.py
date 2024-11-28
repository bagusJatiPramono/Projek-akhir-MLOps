import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error

# Specify the run_id and model path
run_id = "0c58263bf1a04c178feffa0af6cb7c1e"
artifact_path = "model" 

# Load the model
model_uri = f"runs:/{run_id}/{artifact_path}"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Test the model
test_data = [[2.7310e-02, 0.0000e+00, 7.0700e+00, 0.0000e+00, 4.6900e-01,
       6.4210e+00, 7.8900e+01, 4.9671e+00, 2.0000e+00, 2.4200e+02,
       1.7800e+01, 3.9690e+02, 9.1400e+00]]  # Example input
target = [21.6]
predictions = loaded_model.predict(test_data)
mse = mean_squared_error(target,predictions)
print("Predictions:", predictions)

with mlflow.start_run(run_id=run_id):  
    mlflow.log_metric("mse",mse,1)
print("Accuracy logged:", mse)
