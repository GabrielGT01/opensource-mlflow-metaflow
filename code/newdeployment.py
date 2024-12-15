
# | filename: script.py
# | code-line-numbers: true

import logging
import os
from common import PYTHON, FlowMixin, configure_logging, packages
from metaflow import (
    FlowSpec,
    Parameter,
    environment,
    project,
    pypi_base,
    step,
)
from sagemaker import get_boto3_client

configure_logging()

@project(name="weather_report_kaggle")
@pypi_base(
    python=PYTHON,
    packages=packages("mlflow", "boto3", "azure-ai-ml", "azureml-mlflow"),
)
class Deployment(FlowSpec, FlowMixin):
    """Deployment pipeline for deploying a model to AWS SageMaker."""

    endpoint = Parameter("endpoint", help="SageMaker endpoint name.", default="daily_weather")
    target = Parameter("target", help="Deployment target platform.", default="sagemaker")
    data_capture_destination_uri = Parameter(
        "data-capture-destination-uri",
        help="S3 URI for captured data.",
        required=False,
    )
    region = Parameter("region", help="AWS region for deployment.", default="us-east-1")
    assume_role = Parameter(
        "assume-role",
        help="Role to assume for SageMaker deployment.",
        required=False,
    )

    @environment(
        vars={
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI", "http://54.164.38.69:5000"
            ),
        },
    )
    @step
    def start(self):
        """Start the deployment pipeline."""
        import mlflow

        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        logging.info("MLflow tracking URI: %s", self.mlflow_tracking_uri)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        if self.target not in ["sagemaker", "azure"]:
            raise ValueError("Unsupported target platform: %s" % self.target)

        self.model_version = "2.17.2"  # Ensure this matches your ECR repository tag
        self.data = self.load_dataset()
        self.latest_model = self._get_latest_model_from_registry()
        self.next(self.deployment)

    @step
    def deployment(self):
        """Deploy the model to the specified target platform."""
        import tempfile
        from pathlib import Path
        import mlflow

        with tempfile.TemporaryDirectory() as directory:
            mlflow.artifacts.download_artifacts(
                run_id=self.latest_model.run_id, dst_path=directory
            )
            self.model_artifacts = f"file://{(Path(directory) / 'model').as_posix()}"
            logging.info("Model artifacts downloaded to %s", self.model_artifacts)

            if self.target == "sagemaker":
                self._deploy_to_sagemaker()

        self.next(self.inference)

    @step
    def inference(self):
        """Test the deployed model with sample data."""
        samples = self.data.sample(n=3).drop(columns=["Rain"]).reset_index(drop=True)
        if self.target == "sagemaker":
            self._run_sagemaker_prediction(samples)
        self.next(self.end)

    @step
    def end(self):
        """Finalize the deployment."""
        logging.info("Deployment pipeline completed.")

    def _get_latest_model_from_registry(self):
        """Fetch the latest model version from the registry."""
        from mlflow import MlflowClient

        client = MlflowClient()
        response = client.search_model_versions(
            "name='raining'",
            max_results=1,
            order_by=["last_updated_timestamp DESC"],
        )
        if not response:
            raise RuntimeError("No model versions found for 'raining'.")
        latest_model = response[0]
        logging.info("Model version: %s. Artifacts: %s.", self.model_version, latest_model.source)
        return latest_model

    def _deploy_to_sagemaker(self):
        """Deploy the model to SageMaker."""
        from mlflow.deployments import get_deploy_client
        from mlflow.exceptions import MlflowException

        deployment_configuration = {
            "instance_type": "ml.m4.xlarge",
            "instance_count": 1,
            "synchronous": True,
            "archive": True,
            "tags": {"version": self.model_version},
            "image_uri": f"025066244860.dkr.ecr.us-east-1.amazonaws.com/mlflow-pyfunc:{self.model_version}",
        }
        if self.data_capture_destination_uri:
            deployment_configuration["data_capture_config"] = {
                "EnableCapture": True,
                "InitialSamplingPercentage": 100,
                "DestinationS3Uri": self.data_capture_destination_uri,
            }

        if self.assume_role:
            self.deployment_target_uri = f"sagemaker:/{self.region}/{self.assume_role}"
            deployment_configuration["execution_role_arn"] = self.assume_role
        else:
            self.deployment_target_uri = f"sagemaker:/{self.region}"

        deployment_client = get_deploy_client(self.deployment_target_uri)

        try:
            deployment = deployment_client.get_deployment(self.endpoint)
            if not self._is_sagemaker_model_running(deployment):
                self._update_sagemaker_deployment(deployment_client, deployment_configuration)
        except MlflowException:
            self._create_sagemaker_deployment(deployment_client, deployment_configuration)

    def _is_sagemaker_model_running(self, deployment):
        """Check if the deployed model matches the current version."""
        sagemaker_client = get_boto3_client(service="sagemaker", assume_role=self.assume_role)
        model_arn = sagemaker_client.describe_model(
            ModelName=deployment["ProductionVariants"][0]["VariantName"]
        ).get("ModelArn")
        tags = sagemaker_client.list_tags(ResourceArn=model_arn).get("Tags", [])
        return any(tag["Value"] == self.model_version for tag in tags if tag["Key"] == "version")

    def _create_sagemaker_deployment(self, deployment_client, config):
        """Create a new SageMaker deployment."""
        deployment_client.create_deployment(
            name=self.endpoint, model_uri=self.model_artifacts, flavor="python_function", config=config
        )

    def _update_sagemaker_deployment(self, deployment_client, config):
        """Update an existing SageMaker deployment."""
        deployment_client.update_deployment(
            name=self.endpoint, model_uri=self.model_artifacts, flavor="python_function", config=config
        )

    def _run_sagemaker_prediction(self, samples):
        """Run predictions using the deployed SageMaker model."""
        from mlflow.deployments import get_deploy_client
        import pandas as pd

        deployment_client = get_deploy_client(self.deployment_target_uri)
        response = deployment_client.predict(self.endpoint, samples)
        predictions = pd.DataFrame(response["predictions"])[["prediction", "confidence"]]
        logging.info("Predictions:\n%s", predictions)


if __name__ == "__main__":
    Deployment()
