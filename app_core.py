"""Data loading, feature engineering, and the train/test split.

Shared by `app.py` (serving) and `train.py` (offline training) so the two cannot
drift apart. If feature engineering lived in both files, one could be edited
without the other and the app would score inputs the model was never fitted on --
the standard train/serve skew bug, and a silent one.

No Streamlit import here on purpose: this module has to be usable from a plain
script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
ARTIFACT_PATH = ROOT / "artifacts.joblib"

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"

COLUMNS = ["ID", "Product_ID", "Type", "Air_Temp", "Process_Temp", "Rot_Speed",
           "Torque", "Tool_Wear", "Machine_Failure", "TWF", "HDF", "PWF", "OSF", "RNF"]

FEATURES = [
    "Air_Temp", "Process_Temp", "Rot_Speed", "Torque", "Tool_Wear",
    "Temp_Diff", "Power", "Wear_Strain", "Speed_Torque_Ratio",
    "Thermal_Stress", "Type_Encoded",
]

FAILURE_MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]

TYPE_MAP = {"L": 0, "M": 1, "H": 2}


def load_data() -> pd.DataFrame:
    """Read the committed copy; fall back to UCI only if it is missing.

    The CSV is committed so the deployed app has no runtime network dependency --
    a UCI outage would otherwise take the live demo down with it.
    """
    df = pd.read_csv(DATA_PATH if DATA_PATH.exists() else DATA_URL)
    df.columns = COLUMNS
    return df


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # Thermal gradient — drives HDF (Heat Dissipation Failure)
    # If process temp rises relative to air temp, cooling is inadequate
    df["Temp_Diff"] = df["Process_Temp"] - df["Air_Temp"]

    # Mechanical power (W) — proxy for electrical load on motor
    # P = τ × ω; high power + high wear = PWF territory
    df["Power"] = df["Torque"] * (df["Rot_Speed"] * 2 * np.pi / 60)  # SI units

    # Wear-normalized torque — how hard is the tool working per wear-minute?
    # Sudden torque spikes on a worn tool signal imminent TWF
    df["Wear_Strain"] = df["Tool_Wear"] * df["Torque"]

    # Speed-torque ratio — machines operating outside the stable envelope
    # (low speed, high torque) are prone to OSF
    df["Speed_Torque_Ratio"] = df["Rot_Speed"] / (df["Torque"] + 1e-9)

    # Thermal stress index — compound indicator combining temp diff and power
    df["Thermal_Stress"] = df["Temp_Diff"] * df["Power"] / 1e6  # scaled

    # Product type encoding (L < M < H quality tiers have different tolerances)
    df["Type_Encoded"] = df["Type"].map(TYPE_MAP)

    return df


def build_input_row(air, proc, speed, torq, wear, prod_type) -> pd.DataFrame:
    """Build one row for scoring, through the same engineering path as training.

    Routing live input through `engineer_features` rather than restating the
    formulas keeps a slider value and a training row definitionally identical.
    """
    raw = pd.DataFrame([{
        "ID": 0, "Product_ID": "live", "Type": prod_type,
        "Air_Temp": air, "Process_Temp": proc, "Rot_Speed": speed,
        "Torque": torq, "Tool_Wear": wear, "Machine_Failure": 0,
        "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
    }])
    return engineer_features(raw)[FEATURES]


def prepare_data():
    df = engineer_features(load_data())
    X, y = df[FEATURES], df["Machine_Failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return (df, X_train, X_test, X, y, y_train, y_test,
            X_train_s, X_test_s, scaler)
