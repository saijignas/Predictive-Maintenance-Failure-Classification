"""Domain-informed feature engineering for the AI4I 2020 dataset.

The extra features mirror the actual generative rules published for this
dataset (Matzka, 2020) rather than being arbitrary transforms:

- power failure fires when electrical power leaves a safe band
  (power = torque * rotational speed, in rad/s)
- heat dissipation failure fires when the air/process temperature gap is
  too small AND the tool is running slow
- overstrain failure fires when tool_wear * torque exceeds a threshold
  that depends on the product quality variant (L/M/H)

Building these explicitly (instead of only feeding raw sensor columns to
a model) is what lets the explainability step later check whether the
model actually learned the real mechanism or just found a shortcut.
"""
import numpy as np
import pandas as pd

RAW_NUMERIC = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Columns that are themselves failure-cause flags, not sensor readings
# available before a failure happens. Machine failure = OR of these, so
# including them as predictors would leak the label. Dropped everywhere.
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]


def load_raw(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def engineer_features(df):
    out = pd.DataFrame(index=df.index)

    out["air_temp"] = df["Air temperature [K]"]
    out["process_temp"] = df["Process temperature [K]"]
    out["rotational_speed"] = df["Rotational speed [rpm]"]
    out["torque"] = df["Torque [Nm]"]
    out["tool_wear"] = df["Tool wear [min]"]

    out["temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    out["power_w"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"] * (2 * np.pi / 60)
    out["wear_torque"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    type_dummies = pd.get_dummies(df["Type"], prefix="type")
    out = pd.concat([out, type_dummies], axis=1)

    target = df["Machine failure"].astype(int)
    return out, target
