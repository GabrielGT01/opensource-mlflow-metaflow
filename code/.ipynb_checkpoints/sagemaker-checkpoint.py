import json
import boto3
import pandas as pd

def get_boto3_client(service, assume_role=None):
    """Return a Boto3 client for the specified service.

    If the `assume_role` parameter is provided, this function will assume the role and
    return a new client with temporary credentials.
    """
    if not assume_role:
        return boto3.client(service)

    # If we have to assume a role, we need to create a new
    # Security Token Service (STS)
    sts_client = boto3.client("sts")

    # Let's use the STS client to assume the role and return
    # temporary credentials
    credentials = sts_client.assume_role(
        RoleArn=assume_role,
        RoleSessionName="mlschool-session",
    )["Credentials"]

    # We can use the temporary credentials to create a new session
    # from where to create the client for the target service.
    session = boto3.Session(
        aws_access_key_id=credentials["AccessKeyId"],
        aws_secret_access_key=credentials["SecretAccessKey"],
        aws_session_token=credentials["SessionToken"],
    )

    return session.client(service)

def load_labeled_data(s3_client, data_uri, ground_truth_uri):
    """Load any labeled data from the specified S3 location.

    This function will load the data captured from the endpoint during inference that
    has a corresponding ground truth information.
    """
    data = _load_collected_data(s3_client, data_uri, ground_truth_uri)
    return data if data.empty else data[data["Rain"].notna()]

def load_unlabeled_data(s3_client, data_uri, ground_truth_uri):
    """Load any unlabeled data from the specified S3 location.

    This function will load the data captured from the endpoint during inference that
    does not have a corresponding ground truth information.
    """
    data = _load_collected_data(s3_client, data_uri, ground_truth_uri)
    return data if data.empty else data[data["Rain"].isna()]

def _load_collected_data(s3_client, data_uri, ground_truth_uri):
    """Load the data capture from the endpoint and merge it with its ground truth."""
    # data_uri: data or sample batch originating from users or simulated
    data = _load_collected_data_files(s3_client, data_uri)
    ground_truth = _load_ground_truth_files(s3_client, ground_truth_uri)

    if len(data) == 0:
        return pd.DataFrame()

    if len(ground_truth) > 0:
        ground_truth = ground_truth.explode("Rain")
        data["index"] = data.groupby("event_id").cumcount()
        ground_truth["index"] = ground_truth.groupby("event_id").cumcount()

        data = data.merge(
            ground_truth,
            on=["event_id", "index"],
            how="left",
        )
        data = data.rename(columns={"Rain_y": "Rain"}).drop(
            columns=["Rain_x", "index"],
        )

    return data

def _load_ground_truth_files(s3_client, ground_truth_s3_uri):
    """Load the ground truth data from the specified S3 location."""

    def process(row):
        data = row["groundTruthData"]["data"]
        event_id = row["eventMetadata"]["eventId"]

        return pd.DataFrame({"event_id": [event_id], "Rain": [data]})

    df = _load_files(s3_client, ground_truth_s3_uri)

    if df is None:
        return pd.DataFrame()

    processed_dfs = [process(row) for _, row in df.iterrows()]

    return pd.concat(processed_dfs, ignore_index=True)

def _load_collected_data_files(s3_client, data_uri):
    """Load the data captured from the endpoint during inference."""
    # data_uri: data or sample batch originating from users or simulated
    # Imagine S3 files with this structure:
    # File contents (capture-1.jsonl):
    # {
    #    "eventMetadata": {
    #        "inferenceTime": "2024-01-15T10:30:45Z",
    #        "eventId": "unique-event-123456"
    #    },
    #    "captureData": {
    #        "endpointInput": {
    #           "data": "{\"inputs\": [{\"temperature\": 25, \"humidity\": 60}]}"
    #        },
    #        "endpointOutput": {
    #            "data": "{\"predictions\": [{\"rain or no_rain\": 0.75}]}"
    #        }
    #    }
    # }

    def process_row(row):
        # Example: "2024-01-15T10:30:45Z"
        date = row["eventMetadata"]["inferenceTime"]
        # Example: "unique-event-123456" uuid
        event_id = row["eventMetadata"]["eventId"]
        # data sent in to endpoint
        input_data = json.loads(row["captureData"]["endpointInput"]["data"])
        # data sent out via output
        output_data = json.loads(row["captureData"]["endpointOutput"]["data"])

        df = pd.concat(
            [
                (
                    pd.DataFrame(input_data["inputs"])
                    if "inputs" in input_data
                    else pd.DataFrame(
                        input_data["dataframe_split"]["data"],
                        columns=input_data["dataframe_split"]["columns"],
                    )
                ),
                pd.DataFrame(output_data["predictions"]),
            ],
            axis=1,
        )

        df["date"] = date
        df["event_id"] = event_id
        # ground_truth placeholder
        df["Rain"] = None
        return df

    # Read each line
    df = _load_files(s3_client, data_uri)

    if df is None:
        return pd.DataFrame()

    processed_dfs = [process_row(row) for _, row in df.iterrows()]

    # Concatenate all processed DataFrames
    result_df = pd.concat(processed_dfs, ignore_index=True)
    return result_df.sort_values(by="date", ascending=False).reset_index(drop=True)

def _load_files(s3_client, s3_uri):
    """Load every file stored in the supplied S3 location.

    This function will recursively return the contents of every file stored under the
    specified location. The function assumes that the files are stored in JSON Lines
    format.
    """
    # Split bucket name here
    # name of bucket = mlschool
    bucket = s3_uri.split("/")[2]

    # Can give access to folders and subfolders after mlschool
    # join the folders and all with /
    # datastore/groundtruth
    # Imagine we have files like:
    # s3://my-ml-data-bucket/model-captures/2024/capture-1.jsonl
    # s3://my-ml-data-bucket/model-captures/2024/capture-2.jsonl

    # prefix = model-captures/2024/capture-1.jsonl
    prefix = "/".join(s3_uri.split("/")[3:])

    # paginator allows to read each file as a list
    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    files = [
        obj["Key"] for page in pages if "Contents" in page for obj in page["Contents"]
    ]

    if len(files) == 0:
        return None

    # files should return
    # s3://my-ml-data-bucket/model-captures/2024/capture-1.jsonl
    # s3://my-ml-data-bucket/model-captures/2024/capture-2.jsonl

    dfs = []
    for file in files:
        obj = s3_client.get_object(Bucket=bucket, Key=file)
        # Content of capture-1.jsonl:
        # {"Temperature": 25.5, "Humidity": 80, "Prediction": 0.7}
        # {"Temperature": 26.1, "Humidity": 82, "Prediction": 0.6}

        # Content of capture-2.jsonl:
        # {"Temperature": 24.9, "Humidity": 75, "Prediction": 0.5}
        # {"Temperature": 23.7, "Humidity": 70, "Prediction": 0.4}
        data = obj["Body"].read().decode("utf-8")

        json_lines = data.splitlines()

        # Parse each line as a JSON object and collect into a list
        dfs.append(pd.DataFrame([json.loads(line) for line in json_lines]))

    # Concatenate all DataFrames into a single DataFrame
    return pd.concat(dfs, ignore_index=True)