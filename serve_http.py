"""
Script for serving.
"""
import json
import pickle
from datetime import datetime, timedelta
from os import getenv
from threading import Timer

import requests
import numpy as np
from bedrock_client.bedrock.metrics.service import ModelMonitoringService
from google.cloud.storage import Client
from flask import Flask, Response, current_app, request
from werkzeug.utils import secure_filename

from utils.constants import AREA_CODES, STATES, SUBSCRIBER_FEATURES

OUTPUT_MODEL_NAME = "/artefact/lgb_model.pkl"
GCS_CLIENT = Client()


def predict_prob(subscriber_features,
                 model=pickle.load(open(OUTPUT_MODEL_NAME, "rb"))):
    """Predict churn probability given subscriber_features.

    Args:
        subscriber_features (dict)
        model

    Returns:
        churn_prob (float): churn probability
    """
    row_feats = list()
    for col in SUBSCRIBER_FEATURES:
        row_feats.append(subscriber_features[col])

    for area_code in AREA_CODES:
        if subscriber_features["Area_Code"] == area_code:
            row_feats.append(1)
        else:
            row_feats.append(0)

    for state in STATES:
        if subscriber_features["State"] == state:
            row_feats.append(1)
        else:
            row_feats.append(0)

    # Score
    churn_prob = (
        model
        .predict_proba(np.array(row_feats).reshape(1, -1))[:, 1]
        .item()
    )

    # Log the prediction
    pid = current_app.monitor.log_prediction(
        request_body=json.dumps(subscriber_features),
        features=row_feats,
        output=churn_prob
    )

    return churn_prob, pid


# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/", methods=["POST"])
def get_churn():
    """Returns the `churn_prob` given the subscriber features"""

    subscriber_features = request.json
    prob, pid = predict_prob(subscriber_features)
    result = {
        "churn_prob": prob,
        "prediction_id": pid
    }
    return result


@app.before_first_request
def init_background_threads():
    """Global objects with daemon threads will be stopped by gunicorn --preload flag.
    So instantiate them here instead.
    """
    current_app.monitor = ModelMonitoringService()


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Returns real time feature values recorded by prometheus
    """
    body, content_type = current_app.monitor.export_http(
        params=request.args.to_dict(flat=False),
        headers=request.headers,
    )
    return Response(body, content_type=content_type)


@app.route("/labels", methods=["POST"])
def upload_labels():
    """Accepts ground truth label uploads
    """
    f = request.files.get("file")
    if not f or not f.filename:
        return "No file specified", 400

    # Upload labels to GCS / S3 bucket
    bucket = GCS_CLIENT.bucket("span-logs-prediction-staging")
    name = secure_filename(f.filename)
    # TODO: Add server id prefix
    blob = bucket.blob(f"models/labels/pending/{name}")
    blob.upload_from_string(data=f.read())

    def join_label():
        submitter = getenv("BEDROCK_SERVER_ID") or "localhost"
        access_token = getenv("BDRK_API_TOKEN", default="")
        project_id = "default"
        pipeline_public_id = "join-label-24d009e5"
        environment_id = "canary-dev"
        model_artefact_id = "3de3115d-fd76-49a3-91af-5758fdce001e"
        bucket = blob.name
        resp = requests.post(
            f"https://api.amoy.ai/v1/batch_scoring_pipeline/{pipeline_public_id}/run/",
            headers={
                "X-Bedrock-Access-Token": access_token
            },
            json={
                "environment_id": environment_id,
                "model_artefact_id": model_artefact_id,
                "source": {"commit": "join-label"},
                "script_parameters": {
                    "BDRK_LABEL_BUCKET": bucket,
                    "BDRK_POD_NAME": submitter,
                }
            },
            params={"project_id": project_id},
        )
        resp.raise_for_status()
        print(resp.text)

    # Schedule a background task to run label joiner after predictions are logged
    wait = 15 - datetime.utcnow().minute % 15 + 2
    t = Timer(interval=wait * 60, function=join_label)
    t.start()

    return blob.public_url


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()
