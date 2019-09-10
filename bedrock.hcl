version = "1.0"

train {
    image = "basisai/workload-standard:v0.1.2"
    install = ["pip3 install -r requirements.txt"]
    script = [
        {spark-submit {
            script = "train.py"
            // to be passed in as --conf key=value
            conf {
                spark.kubernetes.container.image = "basisai/workload-standard:v0.1.2"
                spark.kubernetes.pyspark.pythonVersion = "3"
                spark.driver.memory = "4g"
                spark.driver.cores = "2"
                spark.executor.instances = "2"
                spark.executor.memory = "4g"
                spark.executor.cores = "2"
                spark.memory.fraction = "0.5"
                spark.sql.parquet.compression.codec = "gzip"
                spark.hadoop.fs.AbstractFileSystem.gs.impl = "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS"
                spark.hadoop.google.cloud.auth.service.account.enable = "true"
            }
            // to be passed in as --key=value
            settings {
                jars = "gs://spark-lib/bigquery/spark-bigquery-latest.jar"
            }
        }}
    ]

    parameters {
        RAW_BIGQUERY_PROJECT = "span-production"
        RAW_BIGQUERY_DATASET = "churn"
        RAW_SUBSCRIBER_TABLE = "subscribers"
        RAW_ALL_CALLS_TABLE = "all_calls"
        LR = "0.05"
        NUM_LEAVES = "10"
        N_ESTIMATORS = "250"
        OUTPUT_MODEL_NAME = "lgb_model.pkl"
    }
}

batch_score {
    image = "basisai/workload-standard:v0.1.2"
    install = ["pip3 install -r requirements.txt && pip3 install pandas-gbq"]
    script = [
        {spark-submit {
            script = "batch_score.py"
            // to be passed in as --conf key=value
            conf {
                spark.kubernetes.container.image = "basisai/workload-standard:v0.1.2"
                spark.kubernetes.pyspark.pythonVersion = "3"
                spark.driver.memory = "4g"
                spark.driver.cores = "2"
                spark.executor.instances = "2"
                spark.executor.memory = "4g"
                spark.executor.cores = "2"
                spark.memory.fraction = "0.5"
                spark.sql.parquet.compression.codec = "gzip"
                spark.hadoop.fs.AbstractFileSystem.gs.impl = "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS"
                spark.hadoop.google.cloud.auth.service.account.enable = "true"
            }
            // to be passed in as --key=value
            settings {
                jars = "gs://spark-lib/bigquery/spark-bigquery-latest.jar"
            }
        }}
    ]

    parameters {
        RAW_BIGQUERY_PROJECT = "span-production"
        RAW_BIGQUERY_DATASET = "churn"
        RAW_SUBSCRIBER_TABLE = "subscribers"
        RAW_ALL_CALLS_TABLE = "all_calls"
        DEST_SUBSCRIBER_SCORE_TABLE = "subscriber_score"
        OUTPUT_MODEL_NAME = "lgb_model.pkl"
    }
}

serve {
    image = "python:3.7"
    install = ["pip3 install -r requirements-serve.txt"]
    script = ["python3 serve_http.py"]
}
