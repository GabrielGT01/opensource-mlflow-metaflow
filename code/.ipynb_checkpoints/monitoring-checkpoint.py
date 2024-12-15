
import logging
import sqlite3
from evidently import ColumnMapping


from common import PYTHON, FlowMixin, configure_logging, packages
from metaflow import (
    FlowSpec,
    Parameter,
    card,
    project,
    pypi_base,
    step,
)
from sagemaker import get_boto3_client, load_labeled_data

configure_logging()


@project(name="weather_report_kaggle")
@pypi_base(
    python=PYTHON,
    packages=packages("evidently", "pandas", "boto3"),
)


class Monitoring(FlowSpec, FlowMixin):
    """A monitoring pipeline to monitor the performance of a hosted model.

    This pipeline runs a series of tests and generates several reports using the
    data captured by the hosted model and a reference dataset.
    """

    datastore_uri = Parameter(
        "datastore-uri",
        help=(
            "The location where the production data is stored. The pipeline supports "
            "loading the data from a SQLite database or from an S3 location that "
            "follows SageMaker's format for capturing data."
        ),
        required=True,
    )

    ground_truth_uri = Parameter(
        "ground-truth-uri",
        help=(
            "The S3 location where the ground truth labels associated with the "
            "endpoint's collected data is stored. The content of this S3 location must "
            "follow SageMaker's format for storing ground truth data."
        ),
        required=False,
    )

    assume_role = Parameter(
        "assume-role",
        help=(
            "The role the pipeline will assume to access the production data in S3. "
            "This parameter is required when the pipeline is running under a set of "
            "credentials that don't have access to the S3 location where the "
            "production data is stored."
        ),
        required=False,
    )

    samples = Parameter(
        "samples",
        help=(
            "The maximum number of samples that will be loaded from the production "
            "datastore to run the monitoring tests and reports. The flow will load "
            "the most recent samples."
        ),
        default=200,
    )

    @card
    @step
    def start(self):
        """Start the monitoring pipeline."""
        from evidently import ColumnMapping
        #evidenly requires having a reference data
        self.reference_data = self.load_dataset()

        # When running some of the tests and reports, we need to have a prediction
        # column in the reference data to match the production dataset [prediction column] too
        #assign the prediction to my target from original dataset
        
        self.reference_data["prediction"] = self.reference_data["Rain"]
        #load my stimulated production data
        self.current_data = self._load_production_datastore()
        
        # Some of the tests and reports require labeled data, so we need to filter out
        # the samples that don't have ground truth labels.
        #check the file and remove rows with no ground turuth label
        self.current_data_labeled = self.current_data[
            self.current_data["rain_status"].notna()
        ]
        
        # The target column: This is the dependent variable you’re trying to predict (for supervised learning problems).
        
        #rain for my original datset        
        #Prediction column: The model's predictions, necessary for performance evaluation.
        #prediction is now on ref dataset and now current dataset
        self.column_mapping = ColumnMapping(
            target="Rain",  # Ground truth column in reference dataset
            prediction="prediction",  # Model predictions
        )

        self.next(self.test_suite)



    @card(type="html")
    @step
    def test_suite(self):
        """Run a test suite of pre-built tests.

        This test suite will run a group of pre-built tests to perform structured data
        and model checks.
        """

        from evidently.test_suite import TestSuite
        from evidently.tests import (
            TestColumnsType,
            TestColumnValueMean,
            TestNumberOfColumns,
            TestNumberOfDriftedColumns,
            TestNumberOfDuplicatedColumns,
            TestNumberOfEmptyColumns,
            TestNumberOfEmptyRows,
            TestNumberOfMissingValues,
            TestShareOfMissingValues,
            TestValueList,
        )

        #check mean value for columns and perform other test
        test_suite = TestSuite(
            tests=[
                TestColumnsType(),
                TestNumberOfColumns(),
                TestNumberOfEmptyColumns(),
                TestNumberOfEmptyRows(),
                TestNumberOfDuplicatedColumns(),
                TestNumberOfMissingValues(),
                TestShareOfMissingValues(),
                TestColumnValueMean("Temperature"),
                TestColumnValueMean("Humidity"),
                TestColumnValueMean("Wind_Speed"),
                TestColumnValueMean("Cloud_Cover"),
                TestColumnValueMean("Pressure"),

                # This test will pass only when the number of drifted columns from the
                # specified list is equal to the specified threshold.
                #gt, le, lt, ge also occur
                # eq stands for "equals." It means the test will check whether the number of drifted columns 
                # is exactly equal to the specified value (in this case, 0).
                # If eq=0 is set in the TestNumberOfDriftedColumns function and drift is detected in one or more columns, the test fails. 
                # This indicates that the number of drifted columns is not equal to 0, which violates the condition specified by eq=0.
                TestNumberOfDriftedColumns(
                    columns=[
                        "Temperature",
                        "Humidity",
                        "Wind_Speed",
                        "Cloud_Cover",
                        "Pressure",
                    ],
                    eq=0,
                ),
            ],
        )

        #this test are mainly for data drift, hence do not require the  target column

        columns = ['prediction','rain_status', 'Rain' ]
        reference_data = self.reference_data.copy().drop(columns= ['Rain','prediction' ])
        current_data = self.current_data.copy().drop(columns=['prediction','rain_status' ])

        test_suite.run(
            reference_data=reference_data,
            current_data=current_data,
        )

        self.html = test_suite.get_html()

        self.next(self.data_quality_report)


    @card(type="html")
    @step
    def data_quality_report(self):
        """Generate a report about the quality of the data and any data drift.

        This report will provide detailed feature statistics, feature behavior
        overview of the data, and an evaluation of data drift with respect to the
        reference data. It will perform a side-by-side comparison between the
        reference and the production data.
        """
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.report import Report

        report = Report(
            metrics=[
                DataQualityPreset(),
                # We want to report dataset drift as long as one of the columns has
                # drifted.drift share = Number of Drifted Features/ Total number of feature
                #  i have 5 feature column , if 1 drifted  thats 1/5 =0,2 threshold at 0,2 indicates a colum dfifted
                #if threshold is 0.4, it will be fglagged if atleast 2 column drifted
                DataDriftPreset(drift_share=0.2),
            ],
        )

        # We don't want to compute data drift in the ground truth column, so we need to
        # remove it from the reference and production datasets.
        #prediction columns on both sides remain so i can see how the model has drifted on its prediction
        #check if C class is still dominant or it has change n production
        reference_data = self.reference_data.copy().drop(columns=['Rain',])
        current_data = self.current_data.copy().drop(columns=['rain_status'])

        report.run(
            reference_data=reference_data,
            current_data=current_data,
            column_mapping=self.column_mapping,
        )

        self.html = report.get_html()
        self.next(self.test_accuracy_score)



    @card(type="html")
    @step
    def test_accuracy_score(self):
        """Run a test to check the accuracy score of the model.

        This test will pass only when the accuracy score is greater than or equal to a
        specified threshold.
        """
        #compare model prediction with ground truth labelling
        #also greater than equal 0.8
        from evidently.test_suite import TestSuite
        from evidently.tests import (
            TestAccuracyScore,
        )
        test_suite = TestSuite(
            tests=[TestAccuracyScore(gte=0.85)],
        )

        if not self.current_data_labeled.empty:
            test_suite.run(
                reference_data=None, #only running acc with predefined vs target
                current_data=self.current_data_labeled,
                column_mapping = ColumnMapping(
                                target="rain_status",  # Ground truth labels (manually labeled in production)
                                prediction="prediction",  # Model predictions,
                                pos_label="rain"
                ),
            )

            self.html = test_suite.get_html()
        else:
            self._message("No labeled production data.")

        self.next(self.target_drift_report)


    @card(type="html")
    @step
    def target_drift_report(self):
        """Generate a Target Drift report.

        This report will explore any changes in model predictions with respect to the
        reference data. This will help us understand if the distribution of model
        predictions is different from the distribution of the target in the reference
        dataset.
        """
        from evidently import ColumnMapping
        from evidently.metric_preset import TargetDriftPreset
        from evidently.report import Report  

        report = Report(
            metrics=[
                TargetDriftPreset(),
            ],
        )
        if not self.current_data_labeled.empty:
            report.run(
                reference_data=self.reference_data,
                current_data=self.current_data_labeled,
                # We only want to compute drift for the prediction column, so we need to
                # specify a column mapping without the target column.
                # i have prediction column in both data
                column_mapping=ColumnMapping(prediction="prediction"),
            )
            self.html = report.get_html()
            
        else:
            self._message("No labeled production data.")


        self.next(self.classification_report)


    @card(type="html")
    @step
    def classification_report(self):
        """Generate a Classification report.

        This report will evaluate the quality of a classification model.
        """
        from evidently.metric_preset import ClassificationPreset
        from evidently.report import Report

        report = Report(
            metrics=[ClassificationPreset()],
        )

        if not self.current_data_labeled.empty:
            report.run(
                # The reference data is using the same target column as the prediction, so
                # we don't want to compute the metrics for the reference data to compare
                # them with the production data.
                reference_data=None,
                current_data=self.current_data_labeled,
                # Define column mapping for the classification report
                column_mapping = ColumnMapping(
                    target="rain_status",  # # Ground truth labels in current production data
                    prediction="prediction",  # Model predictions,
                    pos_label="rain"
                ),
            )
            try:
                self.html = report.get_html()
            except Exception:
                logging.exception("Error generating report.")
        else:
            self._message("No labeled production data.")

        self.next(self.end)



    @step
    def end(self):
        """Finish the monitoring flow."""
        logging.info("Finishing monitoring flow.")


    def _load_production_datastore(self):
        """Load the production data from the specified datastore location."""
        data = None
        if self.datastore_uri.startswith("s3://"):
            data = self._load_production_data_from_s3()
        else:
            data = self._load_production_data_from_sqlite()

        logging.info("Loaded %d samples from the production dataset.", len(data))

        return data

    def _load_production_data_from_s3(self):
        """Load the production data from an S3 location."""
        if self.ground_truth_uri is None:
            message = (
                'The "groundtruth-uri" parameter is required when loading the '
                "production data from S3."
            )
            raise RuntimeError(message)

        s3_client = get_boto3_client(service="s3", assume_role=self.assume_role)

        data = load_labeled_data(
            s3_client,
            data_uri=self.datastore_uri,
            ground_truth_uri=self.ground_truth_uri,
        )

        # We need to remove a few columns that are not needed for the monitoring tests.
        return data.drop(columns=["date", "event_id", "confidence"])



    def _load_production_data_from_sqlite(self):
        """Load the production data from a SQLite database."""
        import pandas as pd

        connection = sqlite3.connect(self.datastore_uri)

        query = (
            "SELECT Temperature, Humidity, Wind_Speed, Cloud_Cover, Pressure, "
            "prediction, rain_status FROM data "
            "ORDER BY date DESC LIMIT ?;"
        )
        #will load the last 200 data always,

        # Notice that i use the `samples` parameter to limit the number of
        # samples we are loading from the database.
        data = pd.read_sql_query(query, connection, params=(self.samples,))

        connection.close()


        return data

    
    def _message(self, message):
        """Display a message in the HTML card associated to a step."""
        self.html = message
        logging.info(message)


if __name__ == "__main__":
    Monitoring()
