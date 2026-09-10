"""Loading data: the bundled teaching sets, and whatever CSV the user brings.

Nothing here depends on the network or on scikit-learn — every bundled dataset
ships as a CSV inside ``data/``.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

CATEGORICAL = "categorical"
NUMERIC = "numeric"


@dataclass
class Dataset:
    """A table plus the metadata the rest of the app needs to reason about it."""

    name: str
    df: pd.DataFrame
    target: str
    feature_types: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    task: str = "classification"
    source: str = ""
    notes: List[str] = field(default_factory=list)

    @property
    def features(self) -> List[str]:
        return [c for c in self.df.columns if c != self.target]

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def classes(self) -> List[Any]:
        if self.task == "regression":
            return []
        return sorted({str(v) for v in self.df[self.target].dropna()})

    def missing_counts(self) -> Dict[str, int]:
        return {c: int(self.df[c].isna().sum()) for c in self.df.columns if self.df[c].isna().any()}

    def type_of(self, column: str) -> str:
        return self.feature_types.get(column, CATEGORICAL)

    def numeric_features(self) -> List[str]:
        return [f for f in self.features if self.type_of(f) == NUMERIC]

    def categorical_features(self) -> List[str]:
        return [f for f in self.features if self.type_of(f) == CATEGORICAL]

    def with_features(self, selected: Sequence[str]) -> "Dataset":
        """A copy restricted to a chosen subset of feature columns."""
        keep = [c for c in selected if c in self.df.columns] + [self.target]
        return Dataset(
            name=self.name,
            df=self.df[keep].copy(),
            target=self.target,
            feature_types={k: v for k, v in self.feature_types.items() if k in keep},
            description=self.description,
            task=self.task,
            source=self.source,
            notes=list(self.notes),
        )


# --------------------------------------------------------------------------- #
# Type inference
# --------------------------------------------------------------------------- #
def infer_feature_types(
    df: pd.DataFrame, target: Optional[str] = None, unique_threshold: int = 10
) -> Dict[str, str]:
    """Numeric columns with enough distinct values are 'numeric', everything else
    is 'categorical'.

    A numeric column with only a handful of levels (a 1/2/3 class code, say) is
    far more useful to a decision tree as a set of labels, which is why the
    threshold exists.  The Data page lets the user override any of this.
    """
    types: Dict[str, str] = {}
    for column in df.columns:
        if column == target:
            continue
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > unique_threshold:
            types[column] = NUMERIC
        elif pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 2:
            types[column] = NUMERIC
        else:
            types[column] = CATEGORICAL
    return types


def infer_task(df: pd.DataFrame, target: str, unique_threshold: int = 15) -> str:
    series = df[target]
    if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > unique_threshold:
        return "regression"
    return "classification"


# --------------------------------------------------------------------------- #
# Bundled datasets
# --------------------------------------------------------------------------- #
BUILTIN_SPECS: Dict[str, Dict[str, Any]] = {
    "PlayTennis": {
        "file": "play_tennis.csv",
        "target": "PlayTennis",
        "task": "classification",
        "drop": ["Day"],
        "types": {"Outlook": CATEGORICAL, "Temperature": CATEGORICAL,
                  "Humidity": CATEGORICAL, "Wind": CATEGORICAL},
        "description": "The 14-row example every ID3 lecture starts from. Small enough to "
                       "check every entropy by hand.",
        "source": "Quinlan (1986), via Mitchell, Machine Learning",
    },
    "PlayTennis + Day (ID column)": {
        "file": "play_tennis.csv",
        "target": "PlayTennis",
        "task": "classification",
        "drop": [],
        "types": {"Day": CATEGORICAL, "Outlook": CATEGORICAL, "Temperature": CATEGORICAL,
                  "Humidity": CATEGORICAL, "Wind": CATEGORICAL},
        "description": "The same data with the useless Day identifier left in. Watch ID3 fall "
                       "for it and C4.5 refuse to.",
        "source": "Quinlan (1986)",
    },
    "Weather (numeric)": {
        "file": "weather_numeric.csv",
        "target": "Play",
        "task": "classification",
        "drop": [],
        "types": {"Outlook": CATEGORICAL, "Temperature": NUMERIC,
                  "Humidity": NUMERIC, "Windy": CATEGORICAL},
        "description": "PlayTennis with real temperatures and humidities — the smallest dataset "
                       "that shows C4.5 picking a numeric threshold.",
        "source": "Witten & Frank, Data Mining",
    },
    "Contact Lenses": {
        "file": "lenses.csv",
        "target": "Lenses",
        "task": "classification",
        "drop": [],
        "types": {"Age": CATEGORICAL, "SpectaclePrescrip": CATEGORICAL,
                  "Astigmatism": CATEGORICAL, "TearProdRate": CATEGORICAL},
        "description": "24 rows, 3 classes, all categorical. A slightly bigger hand-checkable tree.",
        "source": "Cendrowska (1987), UCI",
    },
    "Restaurant (WillWait)": {
        "file": "restaurant.csv",
        "target": "WillWait",
        "task": "classification",
        "drop": [],
        "types": {},
        "description": "12 rows and 10 categorical features — lots of competing splits to compare.",
        "source": "Russell & Norvig, AIMA",
    },
    "Iris": {
        "file": "iris.csv",
        "target": "Species",
        "task": "classification",
        "drop": [],
        "types": {"SepalLength": NUMERIC, "SepalWidth": NUMERIC,
                  "PetalLength": NUMERIC, "PetalWidth": NUMERIC},
        "description": "150 rows, 4 numeric features, 3 classes. The dataset for decision "
                       "boundaries — CART's first cut is the famous PetalLength <= 2.45.",
        "source": "Fisher (1936), UCI",
    },
    "Penguins": {
        "file": "penguins.csv",
        "target": "Species",
        "task": "classification",
        "drop": [],
        "types": {"Island": CATEGORICAL, "BillLength": NUMERIC, "BillDepth": NUMERIC,
                  "FlipperLength": NUMERIC, "BodyMass": NUMERIC, "Sex": CATEGORICAL},
        "description": "344 rows mixing numeric and categorical columns, with a few missing values.",
        "source": "Horst et al., palmerpenguins",
    },
    "Titanic (mini)": {
        "file": "titanic_mini.csv",
        "target": "Survived",
        "task": "classification",
        "drop": [],
        "types": {"Pclass": CATEGORICAL, "Sex": CATEGORICAL, "Age": NUMERIC, "SibSp": NUMERIC,
                  "Parch": NUMERIC, "Fare": NUMERIC, "Embarked": CATEGORICAL},
        "description": "250 passengers with 51 missing ages — the set to use when trying C4.5's "
                       "fractional-instance handling of missing values.",
        "source": "Kaggle / seaborn-data",
    },
    "Auto MPG (regression)": {
        "file": "auto_mpg_mini.csv",
        "target": "MPG",
        "task": "regression",
        "drop": [],
        "types": {"Cylinders": NUMERIC, "Displacement": NUMERIC, "Horsepower": NUMERIC,
                  "Weight": NUMERIC, "Acceleration": NUMERIC, "ModelYear": NUMERIC,
                  "Origin": CATEGORICAL},
        "description": "200 cars with a numeric target — the one dataset here for CART regression.",
        "source": "UCI Auto MPG",
    },
}


def list_builtin() -> List[Dict[str, Any]]:
    """Metadata for the dataset picker, without reading every file."""
    catalogue: List[Dict[str, Any]] = []
    for name, spec in BUILTIN_SPECS.items():
        path = DATA_DIR / spec["file"]
        rows = 0
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                rows = max(sum(1 for _ in handle) - 1, 0)
        catalogue.append(
            {
                "name": name,
                "rows": rows,
                "target": spec["target"],
                "task": spec["task"],
                "description": spec["description"],
                "source": spec["source"],
            }
        )
    return catalogue


def load_builtin(name: str) -> Dataset:
    if name not in BUILTIN_SPECS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(BUILTIN_SPECS)}")
    spec = BUILTIN_SPECS[name]
    df = pd.read_csv(DATA_DIR / spec["file"])
    for column in spec.get("drop", []):
        if column in df.columns:
            df = df.drop(columns=[column])

    types = dict(infer_feature_types(df, spec["target"]))
    types.update({k: v for k, v in spec.get("types", {}).items() if k in df.columns})

    return Dataset(
        name=name,
        df=df,
        target=spec["target"],
        feature_types=types,
        description=spec["description"],
        task=spec["task"],
        source=spec["source"],
    )


# --------------------------------------------------------------------------- #
# User uploads
# --------------------------------------------------------------------------- #
class DatasetError(ValueError):
    """Raised with a message meant to be shown directly to the user."""


def read_table(source: Any, filename: str = "uploaded.csv") -> pd.DataFrame:
    """Read a CSV/TSV from a path, bytes, or a Streamlit upload, sniffing the separator."""
    if isinstance(source, (str, Path)):
        raw = Path(source).read_bytes()
        filename = str(source)
    elif isinstance(source, bytes):
        raw = source
    elif hasattr(source, "read"):
        raw = source.read()
        filename = getattr(source, "name", filename)
        if isinstance(raw, str):
            raw = raw.encode("utf-8")
    else:
        raise DatasetError("Unsupported file object — provide a path, bytes, or an uploaded file.")

    if not raw.strip():
        raise DatasetError("The file is empty.")

    text: Optional[str] = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise DatasetError("Could not decode the file — save it as UTF-8 and try again.")

    separator = ","
    header = text.splitlines()[0] if text.splitlines() else ""
    for candidate in (";", "\t", "|"):
        if header.count(candidate) > header.count(separator):
            separator = candidate

    try:
        df = pd.read_csv(io.StringIO(text), sep=separator, skipinitialspace=True)
    except Exception as exc:  # pragma: no cover - pandas messages vary
        raise DatasetError(f"Could not parse '{filename}' as a table: {exc}") from exc

    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise DatasetError("The file parsed to zero rows.")
    if len(df.columns) < 2:
        raise DatasetError(
            f"Only one column was found (separator guessed as '{separator}'). "
            "A decision tree needs at least one feature plus a target."
        )
    return df


def dataset_from_frame(
    df: pd.DataFrame,
    target: str,
    name: str = "Uploaded dataset",
    type_overrides: Optional[Dict[str, str]] = None,
    task: Optional[str] = None,
) -> Dataset:
    """Wrap a DataFrame, applying the user's target choice and type overrides."""
    if target not in df.columns:
        raise DatasetError(f"Target column '{target}' is not in the table.")
    if df[target].isna().all():
        raise DatasetError(f"Target column '{target}' is entirely empty.")

    working = df[df[target].notna()].copy()
    types = infer_feature_types(working, target)
    types.update({k: v for k, v in (type_overrides or {}).items() if k in working.columns})

    dataset = Dataset(
        name=name,
        df=working,
        target=target,
        feature_types=types,
        description=f"{len(working)} rows x {len(working.columns) - 1} features, uploaded by you.",
        task=task or infer_task(working, target),
        source="user upload",
    )
    dropped = len(df) - len(working)
    if dropped:
        dataset.notes.append(f"Dropped {dropped} row(s) with no value in the target column.")
    return dataset


def load_uploaded(
    source: Any,
    target: Optional[str] = None,
    type_overrides: Optional[Dict[str, str]] = None,
    name: Optional[str] = None,
) -> Dataset:
    """Full path from an uploaded file to a ready-to-train :class:`Dataset`."""
    df = read_table(source)
    chosen = target or df.columns[-1]
    label = name or getattr(source, "name", None) or "Uploaded dataset"
    return dataset_from_frame(df, chosen, name=str(label), type_overrides=type_overrides)
