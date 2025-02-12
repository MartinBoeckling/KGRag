export MLFLOW_TRACKING_URI=sqlite:///data/mlflow/mlruns.db
mlflow ui --backend-store-uri sqlite:///data/mlflow/mlruns.db --port 8080