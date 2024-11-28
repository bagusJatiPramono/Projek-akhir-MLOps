# Training script (simplified)
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn


import pandas as pd
import numpy as np
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# Configure MLflow server
# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# mlflow.set_experiment("Iris Experiment")

# data = load_boston()

X = data
y = target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Start an MLflow run
def train_model(alpha):
    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test,predictions)

        mlflow.log_param("alpha",alpha)
        mlflow.log_metric("mse",mse)
        mlflow.sklearn.log_model(model,"model")

        with open("metrics.csv", "w") as f:
            f.write("metric,value\nmse,{mse}")
        mlflow.log_artifact("metrics.csv")

        print(f"Model with alpha={alpha}, MSE={mse}")
train_model(alpha=0.1)
