
# | filename: script.py
# | code-line-numbers: true

# | filename: script.py
# | code-line-numbers: true
import logging
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import joblib
import matplotlib

os.environ['MPLBACKEND'] = 'Agg'  # Set backend through environment variable
import matplotlib
matplotlib.use('Agg')  # Set backend explicitly
import matplotlib.pyplot as plt

from common import (
    PYTHON,
    #early_stopping_rounds,
    scale_pos_weight,
    FlowMixin,
    build_features_transformer,
    build_model,
    build_target_transformer,
    configure_logging,
    packages,
)

from inference import Model

from metaflow import (
    FlowSpec,
    Parameter,
    card,
    current,
    environment,
    project,
    pypi_base,
    resources,
    step,
)

configure_logging()

@project(name="weather_report_kaggle")
@pypi_base(
    python=PYTHON,
    packages=packages(
        "scikit-learn",
        "pandas",
        "numpy",
        "keras",
        "jax[cpu]",
        "boto3",
        "packaging",
        "mlflow",
        "setuptools",
        "python-dotenv",
        "xgboost"
    ),
)

class Training(FlowSpec, FlowMixin):
    """Training pipeline for predicting rainfall conditions."""

    accuracy_threshold = Parameter(
        "accuracy_threshold",
        help="Minimum accuracy threshold required to register the model.",
        default=0.7
    )
    recall_threshold = Parameter(
        "recall_threshold",
        help="Minimum recall threshold required to register the model.",
        default=0.7
    )

    @card
    @environment(
        vars={"MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")},
    )
    @step
    def start(self):
        """Start and prepare the Training pipeline."""
        import mlflow

        self.mlflow_tracking_url = os.getenv("MLFLOW_TRACKING_URI")
        logging.info("MLFLOW_TRACKING_URI %s", self.mlflow_tracking_url)

        mlflow.set_tracking_uri(self.mlflow_tracking_url)
        self.mode = "production" if current.is_production else "development"
        logging.info("Running flow in %s mode.", self.mode)

        self.data = self.load_dataset()
        
        X = self.data.drop(columns=['Rain'])
        y = self.data['Rain']
        
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        val_data = pd.concat([X_val, y_val], axis=1)

        data_dir = Path('data')
        untransformed_dir = data_dir / 'untransformed_dataset'
        untransformed_dir.mkdir(parents=True, exist_ok=True)

        train_data.to_csv(untransformed_dir / 'train.csv', index=False)
        test_data.to_csv(untransformed_dir / 'test.csv', index=False)
        val_data.to_csv(untransformed_dir / 'validation.csv', index=False)

        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        self.columns=X_test.columns


        self.X_untransformed_train = X_train
        self.X_untransformed_val = X_val
        self.X_untransformed_test = X_test
        self.y_untransformed_train = y_train
        self.y_untransformed_val = y_val
        self.y_untransformed_test = y_test

        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_url}."
            raise RuntimeError(message) from e

        self.training_parameters = {
            "scale_pos_weight": scale_pos_weight,
           # "early_stopping_rounds": early_stopping_rounds,
        }

        self.next(self.transform)

    @card
    @step
    def transform(self):
        """Apply transformation to the entire dataset before model training."""
        self.target_transformer = build_target_transformer()
        self.features_transformer = build_features_transformer()

        logging.info("Transforming data...")

        
        self.y_train = self.y_untransformed_train.to_frame()
        self.y_test = self.y_untransformed_test.to_frame()
        self.y_val = self.y_untransformed_val.to_frame()

        self.y_train = pd.DataFrame(self.target_transformer.fit_transform(self.y_train),columns=["rain"])
        self.y_test = pd.DataFrame(self.target_transformer.transform(self.y_test),columns=["rain"])
        self.y_val = pd.DataFrame(self.target_transformer.transform(self.y_val),columns=["rain"])

        self.X_train = pd.DataFrame(self.features_transformer.fit_transform(self.X_untransformed_train), columns=self.columns)
        self.X_test = pd.DataFrame(self.features_transformer.transform(self.X_untransformed_test), columns=self.columns)
        self.X_val = pd.DataFrame(self.features_transformer.transform(self.X_untransformed_val),columns=self.columns)

        transformed_dir = Path('data') / 'transformed_dataset'
        transformed_dir.mkdir(parents=True, exist_ok=True)

        pd.concat([self.X_train, self.y_train], axis=1).to_csv(transformed_dir / 'train.csv', index=False)
        pd.concat([self.X_test, self.y_test], axis=1).to_csv(transformed_dir / 'test.csv', index=False)
        pd.concat([self.X_val, self.y_val], axis=1).to_csv(transformed_dir / 'validation.csv', index=False)

        self.X = pd.concat([self.X_train, self.X_val], axis=0, ignore_index=True)
        self.y = pd.concat([self.y_train, self.y_val], axis=0, ignore_index=True)

        

        print(f"X_data shape: {self.X.shape}")
        print(f"y_data shape: {self.y.shape}")

        self.next(self.cross_validation, self.train_model)

    @card
    @step
    def cross_validation(self):
        """Generate the indices to split the data for the cross-validation process."""
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.folds = list(enumerate(skf.split(self.X, self.y)))
        self.next(self.unpack_fold, foreach="folds")

    @card
    @step
    def unpack_fold(self):
        """Transform the data to build a model during the cross-validation process."""
        self.fold, (self.train_indices, self.test_indices) = self.input
        self.y_train_fold = self.y.iloc[self.train_indices]
        self.y_test_fold = self.y.iloc[self.test_indices]
        self.X_train_fold = self.X.iloc[self.train_indices]
        self.X_test_fold = self.X.iloc[self.test_indices]
        self.next(self.train_fold)

    @card
    @resources(memory=4096)
    @step
    def train_fold(self):
        """Train a model as part of the cross-validation process."""
        import mlflow

        mlflow.set_tracking_uri(self.mlflow_tracking_url)
        with (
            mlflow.start_run(run_id=self.mlflow_run_id),
            mlflow.start_run(run_name=f"cross-validation-fold-{self.fold}", nested=True) as run,
        ):
            self.mlflow_fold_run_id = run.info.run_id
            mlflow.autolog(log_models=False)

            self.model = build_model(scale_pos_weight=6.95)
            self.model.fit(self.X_train_fold, self.y_train_fold, verbose=0, eval_set=[(self.X_test_fold, self.y_test_fold)])
            self.eval_result = self.model.evals_result_

        self.next(self.evaluate_fold)

    @card
    @step
    def evaluate_fold(self):
        """Evaluate the model we created as part of the cross-validation process."""
        import mlflow
        self.mlflow_tracking_uri = self.mlflow_tracking_url

        self.loss = self.eval_result['validation_0']['logloss'][10]
        self.aucpr = self.eval_result['validation_0']['aucpr'][10]
        self.error = self.eval_result['validation_0']['error'][10]
        self.accuracy = 1 - self.error

        mlflow.set_tracking_uri(self.mlflow_tracking_url)
        with mlflow.start_run(run_id=self.mlflow_fold_run_id):
            mlflow.log_metrics({"test_loss": self.loss, "test_accuracy": self.accuracy, "precision_recall": self.aucpr})

        self.next(self.evaluate_model)
        

    @card
    @environment(
        vars={"MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")},
    )
    @step
    def evaluate_model(self, inputs):
        """Evaluate the overall cross-validation process."""
       
        import mlflow
        import numpy as np

        # We need access to the `mlflow_run_id` and `mlflow_tracking_uri` artifacts
        # that we set at the start of the flow, but since we are in a join step, we
        # need to merge the artifacts from the incoming branches to make them
        # available.
        self.merge_artifacts(inputs, include=["mlflow_run_id", "mlflow_tracking_uri"])
        
        # Gather accuracies and losses from each fold
        accuracies = [input.accuracy for input in inputs]
        losses = [input.loss for input in inputs]
        
        # Calculate mean and standard deviation for accuracy and loss
        self.accuracy = np.mean(accuracies)
        self.accuracy_std = np.std(accuracies)
        self.loss = np.mean(losses)
        self.loss_std = np.std(losses)
        
        # Log metrics to MLflow
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics(
                {
                    "cross_validation_accuracy": self.accuracy,
                    "cross_validation_accuracy_std": self.accuracy_std,
                    "cross_validation_loss": self.loss,
                    "cross_validation_loss_std": self.loss_std,
                },
            )
            
        self.next(self.register_model)

    @card
    @resources(memory=4096)
    @step
    def train_model(self):
        """Train the model that will be deployed to production."""
        import mlflow
        import numpy as np
        from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix

        mlflow.set_tracking_uri(self.mlflow_tracking_url)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.autolog(log_models=False)

            self.model = build_model(scale_pos_weight=6.95)
            self.model.fit(self.X, self.y, verbose=0)
            mlflow.log_params(self.training_parameters)

            self.y_pred = self.model.predict(self.X_test)
            self.final_accuracy = accuracy_score(np.array(self.y_test).reshape(-1, 1), self.y_pred)
            self.precision = precision_score(np.array(self.y_test).reshape(-1, 1), self.y_pred)
            self.recall = recall_score(np.array(self.y_test).reshape(-1, 1), self.y_pred)
            self.f1 = f1_score(np.array(self.y_test).reshape(-1, 1), self.y_pred)
            self.conf_matrix = confusion_matrix(np.array(self.y_test).reshape(-1, 1), self.y_pred)

            feature_importance = pd.DataFrame({'feature': self.X.columns, 'importance': self.model.feature_importances_})
            print("\nTop Most Important Features:")
            print(feature_importance.sort_values('importance', ascending=False).head(5))
               # Log the source file to MLflow
            mlflow.log_artifact(__file__, "source")

            mlflow.log_metrics(
                {
                    "final_accuracy": self.final_accuracy,
                    "precision": self.precision,
                    "recall": self.recall,
                },
            )

        self.next(self.register_model)

    @step
    def register_model(self, inputs):
        """Register the model in the Model Registry if it meets accuracy and recall thresholds."""
        import mlflow
        import tempfile

        self.merge_artifacts(inputs)

        if self.final_accuracy >= self.accuracy_threshold and self.recall >= self.recall_threshold:
            logging.info("Registering model...")
            mlflow.set_tracking_uri(self.mlflow_tracking_url)
            with mlflow.start_run(run_id=self.mlflow_run_id), tempfile.TemporaryDirectory() as directory:
                mlflow.pyfunc.log_model(
                    python_model=Model(data_capture=False),
                    registered_model_name="raining",
                    artifact_path="model",
                    code_paths=[(Path(__file__).parent / "inference.py").as_posix()],
                    artifacts=self._get_model_artifacts(directory),
                    pip_requirements=self._get_model_pip_requirements(),
                    signature=self._get_model_signature(),
                    example_no_conversion=True,
                )
        else:
            logging.info("Model did not meet accuracy or recall thresholds. Skipping registration.")

        self.next(self.end)

    @step
    def end(self):
        """End the Training pipeline."""
        logging.info("The pipeline finished successfully.")

    def _get_model_artifacts(self, directory: str):
        model_path = (Path(directory) / "raining.joblib").as_posix()
        joblib.dump(self.model, model_path)

        features_transformer_path = (Path(directory) / "features.joblib").as_posix()
        target_transformer_path = (Path(directory) / "target.joblib").as_posix()
        joblib.dump(self.features_transformer, features_transformer_path)
        joblib.dump(self.target_transformer, target_transformer_path)

        return {
            "model": model_path,
            "features_transformer": features_transformer_path,
            "target_transformer": target_transformer_path,
        }

    def _get_model_signature(self):
        from mlflow.models import infer_signature 

        return infer_signature(
            model_input={
                "Temperature": 19.738713,
                "Humidity": 75.263991,
                "Wind_Speed": 3.976084,
                "Cloud_Cover": 33.399872,
                "Pressure": 1042.225804,
            },
            model_output={"prediction": "Rain", "confidence": 0.90},
            params={"data_capture": False},
        )

    def _get_model_pip_requirements(self):
        """Return the list of required packages to run the model in production."""
        return [
            f"{package}=={version}"
            for package, version in packages(
                "scikit-learn",
                "pandas",
                "numpy",
                "xgboost",
                "keras",
                "jax[cpu]",
            ).items()
        ]


if __name__ == "__main__":
    Training()
