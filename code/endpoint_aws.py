
import logging

from common import PYTHON, FlowMixin, configure_logging, packages
from metaflow import (
    FlowSpec,
    Parameter,
    project,
    pypi_base,
    step,
)
from sagemaker_aws import load_unlabeled_data

configure_logging()


@project(name="weather_report_kaggle")
@pypi_base(
    python=PYTHON,
    packages=packages("pandas", "numpy", "boto3", "requests"),
)
class Endpoint(FlowSpec, FlowMixin):
    """A pipeline for generating traffic or labeling data captured by a model.

    This pipeline can send fake traffic to a hosted model or generate ground truth
    labels using the data captured by the model.
    """

    action = Parameter(
        "action",
        help=(
            "The action you want to perform. The supported actions are 'traffic' for "
            "sending traffic to the endpoint and 'labeling' for labeling the data "
            "captured by the endpoint."
        ),
        default="traffic",
    )

    target = Parameter(
        "target",
        help=(
            "The target platform hosting the model. The supported values are 'local' "
            "for models hosted as a local inference service and 'sagemaker' for models "
            "hosted on a SageMaker endpoint."
        ),
        default="local",
    )

    target_uri = Parameter(
        "target-uri",
        help=(
            "The location where the pipeline will send the fake traffic or generate "
            "ground truth labels. If generating traffic, this parameter will point to "
            "the hosted model. If generating labels, this parameter will point to the "
            "location of the data captured by the model."
        ),
    )

    samples = Parameter(
        "samples",
        help=(
            "The number of samples that will be sent to the hosted model. Samples will "
            "be sent in batches of 10, so you might end up with a few more samples "
            "than the value you set for this parameter."
        ),
        default=200,
        required=False,
    )

    drift = Parameter(
        "drift",
        help=(
            "Whether to introduce drift in the samples sent to the model. This is "
            "useful for testing the monitoring process."
        ),
        default=False,
        required=False,
    )

    ground_truth_uri = Parameter(
        "ground-truth-uri",
        help=(
            "When labeling data captured by a SageMaker endpoint, this parameter "
            "specifies the S3 location where the ground truth labels will be stored. "
        ),
        required=False,
    )

    ground_truth_quality = Parameter(
        "ground-truth-quality",
        help=(
            "This parameter represents how similar the ground truth labels will be "
            "to the predictions generated by the model. Setting this parameter to a "
            "value less than 1.0 will introduce noise in the labels to simulate "
            "inaccurate model predictions."
        ),
        default=0.8,
        required=False,
    )

    @step
    def start(self):
        """Start the pipeline and load the dataset."""
        if self.action not in ["traffic", "labeling"]:
            message = "The specified action is not supported."
            raise RuntimeError(message)

        if self.target not in ["local", "sagemaker"]:
            message = "The specified target is not supported."
            raise RuntimeError(message)

        #this is imported from common.py
        self.data = self.load_dataset()
        #goin to use the original dataset for stimulation

        #next branch
        self.next(self.prepare_data)

    @step
    def prepare_data(self):
        """Prepare the data and introduce drift before submitting it to the model."""
        import numpy as np
        #drop the target column and any rows thats missing a value
        self.data.pop('Rain')
        self.data = self.data.dropna()

        # If we want to introduce drift, we will add random noise 
        # numerical features in the data.
        if self.action == 'traffic' and self.drift:
            #a generator
            rng = np.random.default_rng()
            #applies drift change in each column using values from 1 up to 3 * the standard deviation
            self.data = self.data.apply(lambda col: col + rng.uniform(1, 3 * col.std(), size=len(col)), axis=0)

        self.next(self.traffic)

    @step
    def traffic(self):
        """Prepare the payload and send traffic to the hosted model."""
        import boto3
        import pandas as pd

        #if action is traffic, it runs this step if not,it ignores
        if self.action == "traffic":
            #the number of samples to be sent to the model
            self.dispatched_samples = 0

            try:
                #sagemaker object required for accessing the model
                if self.target == "sagemaker":
                    sagemaker_runtime = boto3.Session().client("sagemaker-runtime")

                while self.dispatched_samples < self.samples:
                    #sample is already designed to be 200
                    #loops continue untill my dispatched_samples becomes bigger than 200
                    payload = {}
                    #take 10 samples out of the data frame
                    batch = self.data.sample(n=10)
                     #convert to json type input
                    payload["inputs"] = [
                        {
                            k: (None if pd.isna(v) else v)
                            for k, v in row.to_dict().items() 
                        }
                        for _, row in batch.iterrows()
                    ]

                    #if target is local
                    if self.target == "local":
                        self._invoke_local_endpoint(payload)
                    #if its off to sagemaker
                    elif self.target == "sagemaker":
                        self._invoke_sagemaker_endpoint(
                            sagemaker_runtime,
                            payload,
                        )
                    #add the len of every batch sent to the dispatched
                    #batch is 10, so add every 10
                    
                    #break out of the while loop once dispatched_sample is bigger than 200 
                    self.dispatched_samples += len(batch)

            except Exception:
                logging.exception("There was an error sending traffic to the endpoint.")

        self.next(self.labeling)

    @step
    def labeling(self):
        #if action is labelling, it runs this ste or ignores
        """Generate ground truth labels for unlabeled data captured by the model."""
        if self.action == "labeling":
            if self.target == "local":
                self.labeled_samples = self._label_sqlite_data()
            elif self.target == "sagemaker":
                self.labeled_samples = self._label_sagemaker_data()

        self.next(self.end)

    @step
    def end(self):
        """End of the pipeline."""
        if self.action == "traffic":
            logging.info(
                "Dispatched %s samples to the hosted model.",
                self.dispatched_samples,
            )
        elif self.action == "labeling":
            logging.info("Labeled %s samples.", self.labeled_samples)

    #supporting function that are notincluded in the metaflow branch but are needded
    def _invoke_local_endpoint(self, payload):
        """Submit the given payload to a local inference service."""
        import json
        import requests
        
        #the inference.py contains a  data_capture params 
        # data_capture  allows us to keep the data send to the model
        payload["params"] = {"data_capture": True}
        predictions = requests.post(
            url=self.target_uri,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=5,
        )

        return predictions.json()

    def _invoke_sagemaker_endpoint(self, sagemaker_runtime, payload):
        """Submit the given payload to a SageMaker endpoint."""
        import json

        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=self.target_uri,
            ContentType="application/json",
            Body=json.dumps(payload),
        )

        return json.loads(response["Body"].read().decode())

    def _get_label(self, prediction):
        """Generate a fake ground truth label for a sample.
        This function will randomly return a ground truth label taking into account the
        prediction quality we want to achieve.
        """
        import random
         
        return (
            prediction
            #self.ground_truth_quality is 0.8
            #returns the true value or the prediction 
            #if random.random generates a number lower than that
            if random.random() < self.ground_truth_quality
            #if its a number higher geerate a false label fro the option
            else random.choice(["no rain", "rain"])
        )

    def _label_sqlite_data(self):
        """Generate ground truth labels for data captured by a local inference service.

        This function loads any unlabeled data from the SQLite database where the data
        was stored by the model and generates fake ground truth labels for it.
        """
        import sqlite3
        import pandas as pd

        #target_uri is the database in use
        connection = sqlite3.connect(self.target_uri)

        # We want to return any unlabeled samples from the database.
        # remember db contains all model input, prediction,confidence, and ground truth which is rainstatus
        # read the database, return columns where groundtruth is zero
        df = pd.read_sql_query("SELECT * FROM data WHERE rain_status IS NULL", connection)
        logging.info("Loaded %s unlabeled samples from the database.", len(df))

        # If there are no unlabeled samples, we don't need to do anything else.
        if df.empty:
            return 0

        for _, row in df.iterrows():
            #pick the uuid of each row
            uuid = row["uuid"]
            #this runs through the data frame prediction column
            #if the random number in the func generates greater than 0.8
            #it changes, if not it keeps same value 
            label = self._get_label(row["prediction"])
            #labbel and uuid might be a pandas series

            # Update the database, using uuid
            #if uuid matches with that of table
            #change the rain status using the label with the uuid location
            update_query = "UPDATE data SET rain_status = ? WHERE uuid = ?"
            connection.execute(update_query, (label, uuid))

        connection.commit()
        connection.close()

        return len(df)

    def _label_sagemaker_data(self):
        """Generate ground truth labels for data captured by a SageMaker endpoint.

        This function loads any unlabeled data from the location where SageMaker stores
        the data captured by the endpoint and generates fake ground truth labels. The
        function stores the labels in the specified S3 location.
        """
        import json
        from datetime import datetime, timezone
        import boto3

        if not self.ground_truth_uri:
            message = "The 'ground-truth-uri' parameter is required."
            raise RuntimeError(message)

        # Let's make sure the ground truth uri ends with a '/'
        ground_truth_uri = self.ground_truth_uri.rstrip("/") + "/"

        s3_client = boto3.client("s3")

        data = load_unlabeled_data(
            s3_client,
            self.target_uri,
            ground_truth_uri,
        )

        logging.info("Loaded %s unlabeled samples from S3.", len(data))

        # If there are no unlabeled samples, we don't need to do anything else.
        if data.empty:
            return 0

        records = []
        for event_id, group in data.groupby("event_id"):
            predictions = []
            for _, row in group.iterrows():
                predictions.append(self._get_label(row["prediction"]))

            record = {
                "groundTruthData": {
                    # For testing purposes, we will generate a random
                    # label for each request.
                    "data": predictions,
                    "encoding": "CSV",
                },
                "eventMetadata": {
                    # This value should match the id of the request
                    # captured by the endpoint.
                    "eventId": event_id,
                },
                "eventVersion": "0",
            }

            records.append(json.dumps(record))

        ground_truth_payload = "\n".join(records)
        upload_time = datetime.now(tz=timezone.utc)
        uri = (
            "/".join(ground_truth_uri.split("/")[3:])
            + f"{upload_time:%Y/%m/%d/%H/%M%S}.jsonl"
        )

        s3_client.put_object(
            Body=ground_truth_payload,
            Bucket=ground_truth_uri.split("/")[2],
            Key=uri,
        )

        return len(data)


if __name__ == "__main__":
    Endpoint()
