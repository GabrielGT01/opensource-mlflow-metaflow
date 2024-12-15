
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
    """Deployment pipeline.

    This pipeline deploys the latest model from the model registry to a target platform
    and runs a few samples through the deployed model to ensure it's working.
    """

    endpoint = Parameter(
        "endpoint",
        help="The endpoint name that will be created in the target platform.",
        default="daily_weather",
    )

    target = Parameter(
        "target",
        help=(
            "The target platform where the pipeline will deploy the model. "
            "Currently, the supported targets are `sagemaker` and `azure`."
        ),
        default="sagemaker",
    )

    data_capture_destination_uri = Parameter(
        "data-capture-destination-uri",
        help=(
            "The S3 location where SageMaker will store the data captured by the "
            "endpoint. If not specified, data capturing will be disabled for the "
            "endpoint."
        ),
        required=False,
    )

    region = Parameter(
        "region",
        help="The region to use for the deployment.",
        default="us-east-1",
    )

    assume_role = Parameter(
        "assume-role",
        help=(
            "The role the pipeline will assume to deploy the model to SageMaker. "
            "This parameter is required when the pipeline is running under a set of "
            "credentials that don't have access to create the required resources "
            "to host the model in SageMaker."
        ),
        required=False,
    )

    @environment(
        vars={
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI",
                #new tracking for mlflow using ec2
                "http://54.89.239.64:5000",
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

        # We want to make sure that the specified target platform is supported by the
        # pipeline.
        if self.target not in ["sagemaker", "azure"]:
            message = (
                f'Target "{self.target}" is not supported. The supported targets are '
                "`sagemaker` and `azure`."
            )
            raise ValueError(message)
        #imported load_dataset from self.data
        self.data = self.load_dataset()

        self.latest_model = self._get_latest_model_from_registry()

        self.next(self.deployment)


    @step
    def deployment(self):
        """Deploy the model to the appropriate target platform."""
        import tempfile
        from pathlib import Path

        import mlflow

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        # Let's download the model artifacts from the model registry to a temporary
        # directory. This is the copy that we'll use to deploy the model.
        #mlflow is downloading from the new url of ec2 where i trained a new model
        with tempfile.TemporaryDirectory() as directory:
        #downloads the  model associated with the run_id
            mlflow.artifacts.download_artifacts(
                run_id=self.latest_model.run_id,
                dst_path=directory,
            )
            self.model_artifacts = f"file://{(Path(directory) / 'model').as_posix()}"
            logging.info("Model artifacts downloaded to %s ", self.model_artifacts)

            if self.target == "sagemaker":
                self._deploy_to_sagemaker()
            

        self.next(self.inference)

    @step
    def inference(self):
        """Run a few samples through the deployed model to make sure it's working."""
        # Let's select a few random samples from the dataset.
        samples = self.data.sample(n=3).drop(columns=["Rain"]).reset_index(drop=True)

        if self.target == "sagemaker":
            self._run_sagemaker_prediction(samples)

        self.next(self.end)

    @step
    def end(self):
        """Finalize the deployment pipeline."""
        logging.info("The End")
        

    def _get_latest_model_from_registry(self):
        """Get the latest model version from the model registry."""
        from mlflow import MlflowClient

        client = MlflowClient()
         #name of model,but only 1, the last created because desc
        response = client.search_model_versions(
            "name='raining'",
            max_results=1,
            order_by=["last_updated_timestamp DESC"],
        )

        if not response:
            message = 'No model versions found registered under the name "raining".'
            raise RuntimeError(message)

        latest_model = response[0]
        logging.info(
            "Model version: %s. Artifacts: %s.",
            latest_model.version,
            latest_model.source,
        )

        return latest_model


    def _deploy_to_sagemaker(self):
        
        """Deploy the model to SageMaker.

        This function creates a new SageMaker model, endpoint configuration, and
        endpoint to serve the latest version of the model.

        If the endpoint already exists, this function will update it with the latest
        version of the model.
        """
        from mlflow.deployments import get_deploy_client
        from mlflow.exceptions import MlflowException
        
        deployment_configuration = {
            "instance_type": "ml.m4.xlarge",
            "instance_count": 1,
            "synchronous": True,
            # We want to archive resources associated with the endpoint that become
            # inactive as the result of updating an existing deployment.
            "archive": True,
            # Notice how we are storing the version number as a tag.
            "tags": {"version": self.latest_model.version},
        }
        # If the data capture destination is defined, we can configure the SageMaker
        # endpoint to capture data.
        if self.data_capture_destination_uri is not None:
            deployment_configuration["data_capture_config"] = {
                "EnableCapture": True,
                #capture all data coming in:100
                "InitialSamplingPercentage": 100,
                #location for keeping incoming data
                "DestinationS3Uri": self.data_capture_destination_uri,
                
                "CaptureOptions": [
                    {"CaptureMode": "Input"},
                    {"CaptureMode": "Output"},
                ],
                "CaptureContentTypeHeader": {
                    "CsvContentTypes": ["text/csv", "application/octect-stream"],
                    "JsonContentTypes": [
                        "application/json",
                        "application/octect-stream",
                    ],
                },
            }

        if self.assume_role:
            self.deployment_target_uri = f"sagemaker:/{self.region}/{self.assume_role}"
            deployment_configuration["execution_role_arn"] = self.assume_role
            
        else:
            self.deployment_target_uri = f"sagemaker:/{self.region}"

        logging.info("Deployment target URI: %s", self.deployment_target_uri)

        #get_deploy_client is part of the MLflow library, interacts mlflow with sage or azure
        #initializes a client that provides methods for managing deployments,
        #an abstraction layer, allowing you to use a unified interface for deploying models and interacting with the deployment environment.
        deployment_client = get_deploy_client(self.deployment_target_uri)

        try:
            # Let's return the deployment with the name of the endpoint we want to
            # create. If the endpoint doesn't exist, this function will raise an
            # exception.
            #deployment is the endpoint
            deployment = deployment_client.get_deployment(self.endpoint)
    
            # We now need to check whether the model we want to deploy is already
            # associated with the endpoint.
            if self._is_sagemaker_model_running(deployment):
                logging.info(
                    'Enpoint "%s" is already running model "%s".',
                    self.endpoint,
                    self.latest_model.version,
                )
        else:
            # If the model we want to deploy is not associated with the endpoint,
            # we need to update the current deployment/endpoint to replace the previous model
            # with the new one.
            self._update_sagemaker_deployment(
                deployment_client,
                deployment_configuration,
            )
        except MlflowException:
            # If the endpoint doesn't exist, we can create a new deployment.
            self._create_sagemaker_deployment(
                deployment_client,
                deployment_configuration,
            )

    def _is_sagemaker_model_running(self, deployment):
        """Check if the model is already running in SageMaker.

        This function will check if the current model is already associated with a
        running SageMaker endpoint.
        """
        sagemaker_client = get_boto3_client(
            service="sagemaker",
            assume_role=self.assume_role,
        )

        # Here, we're assuming there's only one production variant associated with
        # the endpoint. This code will need to be updated if an endpoint could have
        # multiple variants.
        variant = deployment.get("ProductionVariants", [])[0]

        # From the variant, we can get the ARN of the model associated with the
        # endpoint.
        model_arn = sagemaker_client.describe_model(
            ModelName=variant.get("VariantName"),
        ).get("ModelArn")

        # With the model ARN, we can get the tags associated with the model.
        tags = sagemaker_client.list_tags(ResourceArn=model_arn).get("Tags", [])

        # Finally, we can check whether the model has a `version` tag that matches
        # the model version we're trying to deploy.
        model = next(
            (
                tag["Value"]
                for tag in tags
                if (
                    tag["Key"] == "version"
                    and tag["Value"] == self.latest_model.version
                )
            ),
            None,
        )

        return model is not None

    def _create_sagemaker_deployment(self, deployment_client, deployment_configuration):
        """Create a new SageMaker deployment using the supplied configuration."""
        logging.info(
            'Creating endpoint "%s" with model "%s"...',
            self.endpoint,
            self.latest_model.version,
        )

        print("deployment_configuration", deployment_configuration)

        deployment_client.create_deployment(
            name=self.endpoint,
            model_uri=self.model_artifacts,
            flavor="python_function",
            config=deployment_configuration,
        )

    def _update_sagemaker_deployment(self, deployment_client, deployment_configuration):
        """Update an existing SageMaker deployment using the supplied configuration."""
        logging.info(
            'Updating endpoint "%s" with model "%s"...',
            self.endpoint,
            self.latest_model.version,
        )

        # If you wanted to implement a staged rollout, you could extend the deployment
        # configuration with a `mode` parameter with the value
        # `mlflow.sagemaker.DEPLOYMENT_MODE_ADD` to create a new production variant. You
        # can then route some of the traffic to the new variant using the SageMaker SDK.
        deployment_client.update_deployment(
            name=self.endpoint,
            model_uri=self.model_artifacts,
            flavor="python_function",
            config=deployment_configuration,
        )

    def _run_sagemaker_prediction(self, samples):
        import pandas as pd
        from mlflow.deployments import get_deploy_client

        deployment_client = get_deploy_client(self.deployment_target_uri)

        logging.info('Running prediction on "%s"...', self.endpoint)
        response = deployment_client.predict(self.endpoint, samples)
        df = pd.DataFrame(response["predictions"])[["prediction", "confidence"]]

        logging.info("\n%s", df)
            
        
if __name__ == "__main__":
    Deployment()
