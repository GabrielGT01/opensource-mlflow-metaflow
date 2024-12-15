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
    "xgboost":"2.1.2"
}


TRAINING_BATCH_SIZE = 42
TRAINING_EARLY_STOPPING_rROUNDS=10


class FlowMixin:
    """Base class used to share code across multiple pipelines."""

    dataset = IncludeFile
        "weather_report",
        is_text = True,
        help = (
            "the copy of my weather csv file, this file will be included in the flow chains"
            "and will be used whenever the flow is executed in development mode"
        ),
        default = "weather_forecast_data.csv"
    )

    def load_dataset(self):
        """Load and prepare the dataset.

        When running in production mode, this function reads every CSV file available in
        the supplied S3 location and concatenates them into a single dataframe. When
        running in development mode, this function reads the dataset from the supplied
        string parameter.
        """

        import numpy as np
        #dataset here might be a bucket path of s3

        if current.is_production:
            dataset = os.environ.get("DATASET", self.dataset)

            with S3(s3root=dataset) as s3:
                files = s3.get_all()

                logging.info("Found %d file(s) in remote location", len(files))

                raw_data = [pd.read_csv(StringIO(file.text)) for file in files]
                data = pd.concat(raw_data)
        else:
            # When running in development mode, the raw data is passed as a string,
            # so we can convert it to a DataFrame.
            data = pd.read_csv(StringIO(self.dataset))

        # We want to shuffle the dataset. For reproducibility, we can fix the seed value
        # when running in development mode. When running in production mode, we can use
        # the current time as the seed to ensure a different shuffle each time the
        # pipeline is executed.
        seed = int(time.time() * 1000) if current.is_production else 42
        generator = np.random.default_rng(seed=seed)
        data = data.sample(frac=1, random_state=generator)

        logging.info("Loaded dataset with %d samples", len(data))

        return data

def packages(*names: str):
    """Return a dictionary of the specified packages and their version.

    This function is useful to set up the different pipelines while keeping the
    package versions consistent and centralized in a single location.
    """
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
        transformers = [('rain_status', OrdinalEncoder(), ['Rain'])],

    )

def build_features_transformer():
    """Build a Scikit-Learn transformer to preprocess the feature columns."""
    from sklearn.compose import ColumnTransformer, make_column_selector
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_transformer = make_pipeline(
        SimpleImputer(strategy = "mean"),
        StandardScaler(),
        
    )

    return ColumnTransformer(
        "numeric",
        numeric_transformer,
        # We'll apply the numeric transformer to all columns that are not
        # categorical (object).
        make_column_selector(dtype_exclude="object"),
            ),
    )


def build_model():
    """Build and compile an XGboost classifier to predict the weather."""
    from xgboost import XGBClassifier


    model = XGBClassifier(
        early_stopping_rounds=10,
        eval_metric=['logloss', 'auc', 'error'],
        scale_pos_weight=scale_pos_weight
    )

    return model
    
