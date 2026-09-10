"""Shared fixtures: the small teaching datasets, loaded straight from ``data/``."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"


def _frame(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name)


@pytest.fixture
def play_tennis():
    df = _frame("play_tennis.csv")
    features = ["Outlook", "Temperature", "Humidity", "Wind"]
    types = {f: "categorical" for f in features}
    return df, features, types, "PlayTennis"


@pytest.fixture
def play_tennis_with_id():
    df = _frame("play_tennis.csv")
    features = ["Day", "Outlook", "Temperature", "Humidity", "Wind"]
    types = {f: "categorical" for f in features}
    return df, features, types, "PlayTennis"


@pytest.fixture
def weather_numeric():
    df = _frame("weather_numeric.csv")
    features = ["Outlook", "Temperature", "Humidity", "Windy"]
    types = {"Outlook": "categorical", "Temperature": "numeric",
             "Humidity": "numeric", "Windy": "categorical"}
    return df, features, types, "Play"


@pytest.fixture
def iris():
    df = _frame("iris.csv")
    features = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
    return df, features, {f: "numeric" for f in features}, "Species"


@pytest.fixture
def titanic():
    df = _frame("titanic_mini.csv")
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    types = {"Pclass": "categorical", "Sex": "categorical", "Age": "numeric",
             "SibSp": "numeric", "Parch": "numeric", "Fare": "numeric",
             "Embarked": "categorical"}
    return df, features, types, "Survived"


@pytest.fixture
def auto_mpg():
    df = _frame("auto_mpg_mini.csv")
    features = ["Cylinders", "Displacement", "Horsepower", "Weight",
                "Acceleration", "ModelYear", "Origin"]
    types = {f: "numeric" for f in features}
    types["Origin"] = "categorical"
    return df, features, types, "MPG"
