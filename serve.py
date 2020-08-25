"""
Script for serving.
"""
import json
import pickle

import numpy as np
from bedrock_client.bedrock.model import BaseModel

from utils.constants import AREA_CODES, STATES, SUBSCRIBER_FEATURES


class Model:
    def __init__(self):
        with open("/artefact/lgb_model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def pre_process(self, http_body, files=None):
        """Predict churn probability given subscriber_features.

        Args:
            subscriber_features (dict)
            model

        Returns:
            churn_prob (float): churn probability
        """
        subscriber_features = json.loads(http_body)
        row_feats = [subscriber_features[col] for col in SUBSCRIBER_FEATURES]
        # Apply one-hot encoding
        row_feats.extend(
            int(subscriber_features["Area_Code"] == area_code)
            for area_code in AREA_CODES
        )
        row_feats.extend(int(subscriber_features["State"] == state) for state in STATES)
        return np.array([row_feats], dtype=float)

    def predict(self, features):
        return self.model.predict_proba(features)[:, 1].tolist()
