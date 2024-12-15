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
import numpy as np

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
    """Training pipeline.
    
    This pipeline trains, evaluates, and registers a model to predict the condition of rainfall.
    """

    accuracy_threshold = Parameter(
         "accuracy_threshold",
         help=(
            "Minimum accuracy threshold required to register the model at the end of "
            "the pipeline. The model will not be registered if its accuracy is below "
            "this threshold."
         ),
         default=0.7
    )
    recall_threshold = Parameter(
         "recall_threshold",
         help=(
            "Minimum recall threshold required to register the model at the end of "
            "the pipeline. The model will not be registered if its accuracy is below "
            "this threshold."
         ),
         default=0.7
    )

    @card
    @environment(
        vars={
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:5000",
            ),
        },
    )
    @step
    def start(self):
        """Start and prepare the Training pipeline."""

        import mlflow
        import pandas as pd

        # Set MLflow tracking URL and log it
        self.mlflow_tracking_url = os.getenv("MLFLOW_TRACKING_URI")
        logging.info("MLFLOW_TRACKING_URI %s", self.mlflow_tracking_url)

        mlflow.set_tracking_uri(self.mlflow_tracking_url)
        
        # Set the mode based on the environment
        self.mode = "production" if current.is_production else "development"
        logging.info("Running flow in %s mode.", self.mode)

        self.data = self.load_dataset()
        
        # Define features (X) and target (y)
        X = self.data.drop(columns=['Rain'])
        y = self.data['Rain']
        
        # Split data into train, validation, and test sets with stratification
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        # Create combined training dataset
        train_data = pd.concat([X_train, y_train], axis=1)
        # Create combined test dataset
        test_data = pd.concat([X_test, y_test], axis=1)
        #Create combined val dataset
        val_data = pd.concat([X_val, y_val], axis=1)

         #Save to CSV
        data_dir = Path('data')
        untransformed_dir = data_dir / 'untransformed_dataset'
        untransformed_dir.mkdir(parents=True, exist_ok=True)
        
        train_data.to_csv(untransformed_dir / 'train.csv', index=False)
        test_data.to_csv(untransformed_dir / 'test.csv', index=False)
        val_data.to_csv(untransformed_dir / 'validation.csv', index=False)
        
        
        # Print dataset sizes and class distribution in each split
        print("\nDataset sizes:")
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
            # Start an MLFlow run and assign the run ID
            #current is a property of metaflow 
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_url}."
            raise RuntimeError(message) from e


        # This is the configuration we'll use to train the model. We want to set it up
        # at this point so we can reuse it later throughout the flow.
        self.training_parameters = {
            "scale_pos_weight": scale_pos_weight,
           #  "early_stopping_rounds": early_stopping_rounds,
        }


        # Proceed to the next steps in which is transform the data
        self.next(self.transform)



    @card
    @step
    def transform(self):
        
        """Apply the transformation pipeline to the entire dataset.

        This function transforms the columns of the entire dataset before.
        we train the model
        
        We want to store the transformers as artifacts so we can later use them
        to transform the input data during inference and also save the transformed test data.
        """

        # Let's build the SciKit-Learn pipeline to process the target column and use it
        # to transform the data.
        
        
        
        self.target_transformer = build_target_transformer()
        self.features_transformer = build_features_transformer()

        logging.info("Transforming data ...")
        
        self.y_train = self.y_untransformed_train.to_frame()
        self.y_test = self.y_untransformed_test.to_frame()
        self.y_val = self.y_untransformed_val.to_frame()


        # Let's build the SciKit-Learn pipeline to process the feature columns and target column
        # to transform the training.

        self.y_train = pd.DataFrame(self.target_transformer.fit_transform(self.y_train),columns=["rain"])
        self.y_test = pd.DataFrame(self.target_transformer.transform(self.y_test),columns=["rain"])
        self.y_val = pd.DataFrame(self.target_transformer.transform(self.y_val),columns=["rain"])

        self.X_train = pd.DataFrame(self.features_transformer.fit_transform(self.X_untransformed_train), columns=self.columns)
        self.X_test = pd.DataFrame(self.features_transformer.transform(self.X_untransformed_test), columns=self.columns)
        self.X_val = pd.DataFrame(self.features_transformer.transform(self.X_untransformed_val),columns=self.columns)

        transformed_dir = Path('data') / 'transformed_dataset'
        transformed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create combined training dataset
        pd.concat([self.X_train, self.y_train], axis=1).to_csv(transformed_dir / 'train.csv', index=False)
        pd.concat([self.X_test, self.y_test], axis=1).to_csv(transformed_dir / 'test.csv', index=False)
        pd.concat([self.X_val, self.y_val], axis=1).to_csv(transformed_dir / 'validation.csv', index=False)

        #concat all to form a large data
        self.X = pd.concat([self.X_train, self.X_val], axis=0, ignore_index=True)
        self.y = pd.concat([self.y_train, self.y_val], axis=0, ignore_index=True)


        print(f"X_data shape: {self.X.shape}")
        print(f"y_data shape: {self.y.shape}")

        # Now that everything is set up, we want to run a cross-validation process
        # to evaluate the model and train a final model on the entire dataset. Since
        # these two steps are independent, we can run them in parallel.
        self.next(self.cross_validation, self.train_model)



    @card
    @step
    def cross_validation(self):
        """Generate the indices to split the data for the cross-validation process."""

        from sklearn.model_selection import StratifiedKFold
        # We are going to use a 5-fold cross-validation process to evaluate the model,
        # so let's set it up. We'll shuffle the data before splitting it into batches.

        # Set up 5-fold cross-validation with stratification
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Generate the indices for each fold, generates a number 
        # because of enumerate and also the test and train indices, so three output
        self.folds = list(enumerate(skf.split(self.X, self.y)))

        # We want to transform the data and train a model using each fold, so we'll use
        # `foreach` to run every cross-validation iteration in parallel. Notice how we
        # pass the tuple with the fold number and the indices to next step.
        self.next(self.unpack_fold, foreach="folds")


    @card
    @step
    def unpack_fold(self):
        """Transform the data to build a model during the cross-validation process.

        This step will run for each fold in the cross-validation process. It uses
        a SciKit-Learn pipeline to preprocess the dataset before training a model.
        """

        # Let's start by unpacking the indices representing the training and test data
        # for the current fold. We computed these  three values in the previous step and passed
        # them as the input to this step.
        # enumerate and the test,train indices

        self.fold, (self.train_indices, self.test_indices) = self.input

        logging.info("unpacking fold %d...", self.fold)

        # We need to turn the target column into a shape that the Scikit-Learn
        # pipeline understands.

        self.y_train_fold = self.y.iloc[self.train_indices]
        self.y_test_fold = self.y.iloc[self.test_indices]
        self.X_train_fold = self.X.iloc[self.train_indices]
        self.X_test_fold = self.X.iloc[self.test_indices]

        # After processing the data and storing it as artifacts in the flow, we want
        # to train a model.
        
        self.next(self.train_fold)

    @card
    @resources(memory=4096)
    @step
    def train_fold(self):
        """Train a model as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It trains the
        model using the data we processed in the previous step.
        """
        import mlflow

        logging.info("Training fold %d...", self.fold)

        # Let's track the training process under the same experiment we started at the
        # beginning of the flow. Since we are running cross-validation, we can create
        # a nested run for each fold to keep track of each separate model individually.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with (
            #start the run then, put each cross validation run under
            mlflow.start_run(run_id=self.mlflow_run_id),
            #expecting to see fold 0, fold 1 corresponding to the 5 cross validated data
            mlflow.start_run(
                run_name=f"cross-validation-fold-{self.fold}",
                nested=True,
            ) as run,
        ):

            # Let's store the identifier of the nested run in an artifact so we can
            # reuse it later when we evaluate the model from this fold.
            #each nested fold run will create a  run.info.run_id
            self.mlflow_fold_run_id = run.info.run_id
            # Let's configure the autologging for the training process. Since we are
            # training the model corresponding to one of the folds, we won't log the
            # model itself.
            mlflow.autolog(log_models=False)


            # Let's now build and fit the model on the training data. Notice how we are
            # using the training data we processed and stored as artifacts in the
            # `unpacked` step.

            #due to imbalance,calculate the weight balance scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            

            self.model = build_model(scale_pos_weight = 6.95)
            self.model.fit(
                self.X_train_fold, 
                self.y_train_fold, 
                verbose = 0,
                eval_set=[(self.X_test_fold,self.y_test_fold)]
        )

            self.eval_result = self.model.evals_result_ 

        # After training a model for this fold, we want to evaluate it.
        self.next(self.evaluate_fold)


    @card
    @step
    def evaluate_fold(self):
        """Evaluate the model we created as part of the cross-validation process.

        This step will run for each fold in the cross-validation process. It evaluates
        the model using the test data for this fold.
        """
        import mlflow

        logging.info("Evaluating fold %d...", self.fold)
        
        # Let's evaluate the model using the result of  data we processed as eval_set=[(self.x_test,self.y_test)]
        # XGBClassifier. The error metric calculates the misclassification rate, so accuracy is 1 - error.
        ### aucpr (Area Under the Precision-Recall Curve)
        ### Calculates the area under the precision-recall curve. 
        ### Useful for highly imbalanced datasets where AUC might not provide a sufficient distinction.

        
        self.loss = self.eval_result['validation_0']['logloss'][10]
        self.aucpr = self.eval_result['validation_0']['aucpr'][10]
        self.error = self.eval_result['validation_0']['error'][10]

        self.accuracy = 1 - self.error

        logging.info(
            "Fold %d - loss: %f - accuracy: %f - aucpr: %f",
            self.fold,
            self.loss,
            self.accuracy,
            self.aucpr
        )

        # Let's log everything under the same nested run we created when training the
        # current fold's model.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_fold_run_id):
            mlflow.log_metrics(
                {
                    "test_loss": self.loss,
                    "test_accuracy": self.accuracy,
                    "precision_recall":self.aucpr
                },
            )

        # When we finish evaluating every fold in the cross-validation process, we want
        # to evaluate the overall performance of the model by averaging the scores from
        # each fold.
        
        self.next(self.evaluate_model)



    @card
    @step
    def evaluate_model(self, inputs):
        """Evaluate the overall cross-validation process.

        This function averages the score computed for each individual model to
        determine the final model performance.
        """
        import mlflow
        import numpy as np

        # We need access to the `mlflow_run_id` and `mlflow_tracking_uri` artifacts
        # that we set at the start of the flow, but since we are in a join step,
        # remember we are coming from a for each fold,now we need to average the 
        # mean of our action, so this first enables us comibine the result of each fold runs
        #  we need to merge the artifacts from the incoming branches to make them
        # available.
        
        self.merge_artifacts(inputs, include=["mlflow_run_id", "mlflow_tracking_uri"])
        # Let's calculate the mean and standard deviation of the accuracy and loss from
        # all the cross-validation folds. Notice how we are accumulating these values
        # using the `inputs` parameter provided by Metaflow.

        #inputs contain all what ive created and assigned to self
        
        # Gather accuracies and losses from each fold
        accuracies = [input.accuracy for input in inputs]
        losses = [input.loss for input in inputs]
        
        # Calculate mean and standard deviation for accuracy and loss
        self.accuracy = np.mean(accuracies)
        self.accuracy_std = np.std(accuracies)
        self.loss = np.mean(losses)
        self.loss_std = np.std(losses)

        logging.info("Accuracy: %f ±%f", self.accuracy, self.accuracy_std)
        logging.info("Loss: %f ±%f", self.loss, self.loss_std)


        # Let's log the model metrics on the parent run.
        # self.mlflow_run_id was first created at the start
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(run_id=self.mlflow_run_id):
            mlflow.log_metrics(
                {
                    "cross_validation_accuracy": self.accuracy,
                    "cross_validation_accuracy_std": self.accuracy_std,
                    "cross_validation_loss": self.loss,
                    "cross_validation_loss_std": self.loss_std,
                },
            )
        
                                
        
        # After we finish evaluating the cross-validation process, we can send the flow
        # to the registration step to register where we'll register the final version of
        # the model.
        self.next(self.register_model)



    @card
    @resources(memory=4096)
    @step
    def train_model(self):
        """Train the model that will be deployed to production.

        This function will train the model using the entire dataset.
        """
        import numpy as np
        import mlflow
        from sklearn.metrics import (accuracy_score, classification_report, 
                           precision_score, recall_score, f1_score,
                           confusion_matrix)
        
        # Get the tracking URI from the environment variable instead of relying on artifact merger
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    
        # Let's log the training process under the experiment we started at the
        # beginning of the flow.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        # Let's log the training process under the experiment we started at the
        # beginning of the flow.
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        with mlflow.start_run(run_id=self.mlflow_run_id):
            # Let's disable the automatic logging of models during training so we
            # can log the model manually during the registration step.
            mlflow.autolog(log_models=False)

            # Let's now build and fit the model on the entire dataset.
            self.model = build_model(scale_pos_weight = 6.95)
            self.model.fit(
                self.X, 
                self.y, 
                verbose = 0,
            )


            # Let's log the training parameters we used to train the model.
            mlflow.log_params(self.training_parameters)

            # Make predictions
            
            self.y_pred = self.model.predict(self.X_test)
            # Calculate various metrics
            self.final_accuracy = accuracy_score(np.array(self.y_test).reshape(-1, 1), self.y_pred)
            self.precision = precision_score(np.array(self.y_test).reshape(-1, 1), self.y_pred)
            self.recall = recall_score(np.array(self.y_test).reshape(-1, 1), self.y_pred)
            self.f1 = f1_score(np.array(self.y_test).reshape(-1, 1), self.y_pred)
            self.conf_matrix = confusion_matrix(np.array(self.y_test).reshape(-1, 1), self.y_pred)

            # Print detailed results
            print("\nModel Performance Metrics:")
            print(f"Accuracy: {self.final_accuracy:.4f}")
            print(f"Precision: {self.precision:.4f}")
            print(f"Recall: {self.recall:.4f}")
            print(f"F1 Score: {self.f1:.4f}")
            
            print("\nConfusion Matrix:")
            print(self.conf_matrix)
            
            print("\nDetailed Classification Report:")
            print(classification_report(np.array(self.y_test).reshape(-1, 1), self.y_pred))

            feature_importance = pd.DataFrame({'feature': self.X.columns, 'importance': self.model.feature_importances_})
            print("\nTop Most Important Features:")
            print(feature_importance.sort_values('importance', ascending=False).head(5))

            mlflow.log_metrics(
                {
                    "final_accuracy": self.final_accuracy,
                    "precision": self.precision,
                    "recall": self.recall,
                },
            )
        

        # After we finish training the model, we want to register it.
        self.next(self.register_model)

    @step
    def register_model(self, inputs):
        """Register the model in the Model Registry.

        This function will prepare and register the final model in the Model Registry.
        This will be the model that we trained using the entire dataset.

        We'll only register the model if its accuracy is above a predefined threshold.
        """
        import tempfile

        import mlflow

        # Since this is a join step, we need to merge the artifacts from the incoming
        # branches to make them available here.
        self.merge_artifacts(inputs)

        # We only want to register the model if its accuracy is above the threshold
        # specified by the `accuracy_threshold` parameter.
        if self.final_accuracy >= self.accuracy_threshold:
            if self.self.recall >= self.recall_threshold:
                logging.info("Registering model...")

                # We'll register the model under the experiment we started at the beginning
                # of the flow. We also need to create a temporary directory to store the
                # model artifacts.
                mlflow.set_tracking_uri(self.mlflow_tracking_uri)
                with (
                    mlflow.start_run(run_id=self.mlflow_run_id),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    # We can now register the model using the name "raining" in the Model
                    # Registry. This will automatically create a new version of the model.

                    mlflow.pyfunc.log_model(
                    python_model=Model(data_capture=False),
                    registered_model_name="raining",
                    artifact_path="model",
                    code_paths=[(Path(__file__).parent / "inference.py").as_posix()],
                    artifacts=self._get_model_artifacts(directory),
                    pip_requirements=self._get_model_pip_requirements(),
                    signature=self._get_model_signature(),
                    # Our model expects a Python dictionary, so we want to save the
                    # input example directly as it is by setting`example_no_conversion`
                    # to `True`.
                    example_no_conversion=True,
                )



        else:
            logging.info(
                "The accuracy of the model (%.2f) is lower than the accuracy threshold (%.2f) or the recall (%.2f) is lower than the recall threshold"
                "(%.2f). Skipping model registration.",
                self.accuracy,
                self.accuracy_threshold,
                self.recall,
                self.accuracy_threshold
                
            )

        # Let's now move to the final step of the pipeline.
        self.next(self.end)

    @step
    def end(self):
        """End the Training pipeline."""
        logging.info("The pipeline finished successfully.")





    def _get_model_artifacts(self, directory: str):
        """Return the list of artifacts that will be included with model.

        The model must preprocess the raw input data before making a prediction, so we
        need to include the Scikit-Learn transformers as part of the model package.
        """
        import joblib

        # Let's start by saving the model inside the supplied directory.
        #directory is a temp directory created in pyfunc log model
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


        return {
            "model": model_path,
            "features_transformer": features_transformer_path,
            "target_transformer": target_transformer_path,
        }
            


    def _get_model_signature(self):
        """Return the model's signature.

        The signature defines the expected format for model inputs and outputs. This
        definition serves as a uniform interface for appropriate and accurate use of
        a model.
        """
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

        
