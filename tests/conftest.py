import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)  # scripts use repo-relative paths
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import pytest

import dashboard_utils as du


@pytest.fixture(scope="session")
def data():
    return du.load_data()


@pytest.fixture(scope="session")
def matrix(data):
    return du.get_feature_matrix(data["features"])


@pytest.fixture(scope="session")
def models(matrix):
    X = matrix[0]
    return du.train_models(X)


@pytest.fixture(scope="session")
def y_true(data):
    m = data["features"][["user_id"]].merge(data["labels"][["user_id", "is_malicious"]], on="user_id", how="left")
    return m["is_malicious"].fillna(False).astype(int).values
