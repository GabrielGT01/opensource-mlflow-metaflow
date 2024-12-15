# | filename: script.py
# | code-line-numbers: true

import logging
import logging.config
import os
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
from metaflow import S3, IncludeFile, current


PYTHON = "3.12"
PACKAGES = {
    "scikit-learn": "1.5.2",
    "pandas": "2.2.3",
    "numpy": "2.1.1",
    "keras": "3.6.0",
    "jax[cpu]": "0.4.35",
    "boto3": "1.35.32",
    "packaging": "24.1",
    "mlflow": "2.17.1",
    "setuptools": "75.1.0",
    "requests": "2.32.3",
    "evidently": "0.4.33",
    "azure-ai-ml": "1.19.0",
    "azureml-mlflow": "1.57.0.post1",
    "python-dotenv": "1.0.1",
    "xgboost": "2.1.2"
}



class FlowMixin:
    """Base class used to share code across multiple pipelines."""

    # IncludeFile from Metaflow to include local or S3-based files
    dataset = IncludeFile(
        "weather_report",
        is_text=True,
        help=(
            "The local copy of the weather dataset. This file will be included in the "
            "flow and used whenever the flow is executed in development mode."
        ),
        default="weather_forecast_data.csv",
    )

    def load_dataset(self):
        """Load and prepare the dataset.
        
        Loads data from S3 in production mode or locally in development.
        """

        import numpy as np

        if current.is_production:
            dataset = os.environ.get("DATASET", self.dataset)

            with S3(s3root=dataset) as s3:
                files = s3.get_all()

                logging.info("Found %d file(s) in remote location", len(files))

                raw_data = [pd.read_csv(StringIO(file.text)) for file in files]
                data = pd.concat(raw_data)
        else:
            # In development, load from the local dataset string.
            data = pd.read_csv(StringIO(self.dataset))

        # Shuffle the dataset for reproducibility
        seed = int(time.time() * 1000) if current.is_production else  42
        generator = np.random.default_rng(seed=seed)
        data = data.sample(frac=1, random_state=generator)

        logging.info("Loaded dataset with %d samples", len(data))

        return data

def packages(*names: str):
    """Return a dictionary of the specified packages and their version."""

    return {name: PACKAGES[name] for name in names if name in PACKAGES}

def configure_logging():
    """Configure logging handlers and return a logger instance."""
    if Path("logging.conf").exists():
        logging.config.fileConfig("logging.conf")
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )


def build_target_transformer():
    """Build a Scikit-Learn transformer to preprocess the target column."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder

    return ColumnTransformer(
        transformers=[('rain_status', OrdinalEncoder(), ['Rain'])],
    )


def build_features_transformer():
    """Build a Scikit-Learn transformer to preprocess the feature columns."""
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Pipeline for numeric data only
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
    )

    # Apply the numeric transformer to all columns, assuming they are all numeric
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                make_column_selector(dtype_include="number"),  # Selects only numeric columns
            ),
        ],
    )


def build_model(TRAINING_SCALE_POS_WEIGHT):
    """Build and compile an XGBoost classifier to predict the weather."""
    from xgboost import XGBClassifier

    # Set scale_pos_weight for imbalanced classes (example value, adjust as needed)
    

    model = XGBClassifier(
        early_stopping_rounds=10,
        eval_metric=['logloss', 'aucpr', 'error'],
        scale_pos_weight= TRAINING_SCALE_POS_WEIGHT,
        random_state=42
    )

    return model
