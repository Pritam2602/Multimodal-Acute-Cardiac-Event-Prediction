import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("machine_measurements.csv")

# Preview data
print(df.head())
print(df.info())

# Remove unnecessary report columns
columns_to_remove = [
    "report_0",
    "report_1",
    "report_2",
    "report_3",
    "report_4",
    "report_5",
    "report_6",
    "report_7",
    "report_8",
    "report_9",
    "report_10",
    "report_11",
    "report_12",
    "report_13",
    "report_14",
    "report_15",
    "report_16",
    "report_17"
]

df = df.drop(columns_to_remove, axis=1)

print(df.head())
print(df.info())

# -----------------------------
# Feature Engineering
# -----------------------------

# Heart Rate (assuming rr_interval is in milliseconds)
df["heart_rate"] = 60000 / df["rr_interval"]

# PR interval
df["PR_interval"] = df["qrs_onset"] - df["p_onset"]

# QRS duration
df["QRS_duration"] = df["qrs_end"] - df["qrs_onset"]

# QT interval
df["QT_interval"] = df["t_end"] - df["qrs_onset"]

# QTc (Bazett correction)
df["QTc"] = df["QT_interval"] / np.sqrt(df["rr_interval"] / 1000)

# RR interval (rename for consistency)
df["RR_interval"] = df["rr_interval"]

# Axis columns
df["P_axis"] = df["p_axis"]
df["QRS_axis"] = df["qrs_axis"]
df["T_axis"] = df["t_axis"]

# Keep only required columns
df = df[
    [
        "subject_id",
        "study_id",
        "heart_rate",
        "PR_interval",
        "QRS_duration",
        "QT_interval",
        "QTc",
        "P_axis",
        "QRS_axis",
        "T_axis",
        "RR_interval",
    ]
]

print(df.info())
print(df.head())

# Save processed features
df.to_csv("ecg_features.csv", index=False)