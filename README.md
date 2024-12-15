# opensource-mlflow-metaflow

## Rain Prediction Model
- This uses Metaflow designed by Netflix,mlflow for tracking and AWS ECR to host  the model and EC2 for computing, evidently AI to track data and moel drift
- This project implements a machine learning model to predict rainfall conditions. The model is integrated with data capture mechanisms and logging, making it suitable for production use.

## Features

- **Rain Prediction:** Uses a machine learning model to predict whether it will rain.
- **Data Capture:** Stores input data and predictions in SQLite and JSON Lines for monitoring and analysis.
- **Logging:** Configured to provide detailed logs for debugging and tracking.

## Files

1. **`inference.py`**
   - Handles prediction requests.
   - Captures input and output data in SQLite and JSON Lines.
   - Includes a custom MLflow model class.

2. **`training.py`**
   - (Expected) Contains code for training the rain prediction model.

3. **`deployment.py`**
   - Creates the endpoint and endpoint configuration.

## How to Use

### Prerequisites

- Python 3.8+
- Required Python packages:
  - `joblib`
  - `mlflow`
  - `numpy`
  - `pandas`
  - `metaflow`
  - `AWS`
  - `ECR`
 
