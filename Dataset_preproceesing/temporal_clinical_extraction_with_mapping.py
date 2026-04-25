# ==========================================
# TEMPORAL CLINICAL FEATURE EXTRACTION
# WITH d_items & d_labitems MAPPING
# ==========================================

import pandas as pd
from functools import reduce

# ==========================================
# FILE PATHS (UPDATE THESE)
# ==========================================
ADMISSIONS_PATH = "D:\MINI_PROJECT\mimic_data\admissions.csv.gz"
CHARTEVENTS_PATH = "D:\MINI_PROJECT\mimic_data\chartevents.csv.gz"
LABEVENTS_PATH = "D:\MINI_PROJECT\mimic_data\labevents.csv.gz"
RECORD_LIST_PATH = "D:\MINI_PROJECT\mimic_data\record_list.csv.gz"

D_ITEMS_PATH = "D:\MINI_PROJECT\mimic_data\d_items.csv"
D_LABITEMS_PATH = "D:\MINI_PROJECT\mimic_data\d_labitems.csv"

OUTPUT_PATH = "D:\MINI_PROJECT\mimic_data\temporal_clinical_features.parquet"

# ==========================================
# LOAD DATA
# ==========================================
print("Loading datasets...")

admissions = pd.read_csv(ADMISSIONS_PATH)
chartevents = pd.read_csv(CHARTEVENTS_PATH)
labevents = pd.read_csv(LABEVENTS_PATH)
record_list = pd.read_csv(RECORD_LIST_PATH)

d_items = pd.read_csv(D_ITEMS_PATH)
d_labitems = pd.read_csv(D_LABITEMS_PATH)

# ==========================================
# PREPROCESS TIME COLUMNS
# ==========================================
print("Processing time columns...")

admissions["admittime"] = pd.to_datetime(admissions["admittime"])
admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])

record_list["ecg_time"] = pd.to_datetime(record_list["ecg_time"])

chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])
labevents["charttime"] = pd.to_datetime(labevents["charttime"])

# Normalize labels
d_items["label"] = d_items["label"].str.lower()
d_labitems["label"] = d_labitems["label"].str.lower()

# ==========================================
# DEFINE FEATURES (BY NAME)
# ==========================================

VITAL_FEATURES = [
    "heart rate",
    "respiratory rate",
    "blood pressure"
]

LAB_FEATURES = [
    "troponin",
    "creatinine",
    "sodium",
    "potassium",
    "lactate",
    "hemoglobin"
]

# ==========================================
# GET ITEMIDS USING MAPPING TABLES
# ==========================================

def get_itemids(mapping_df, keywords):
    itemids = {}

    for feature in keywords:
        matched = mapping_df[
            mapping_df["label"].str.contains(feature, na=False)
        ]

        ids = matched["itemid"].unique().tolist()
        itemids[feature] = ids

        print(f"{feature} -> {len(ids)} itemids")

    return itemids


print("\nMapping itemids...")
vital_itemids = get_itemids(d_items, VITAL_FEATURES)
lab_itemids = get_itemids(d_labitems, LAB_FEATURES)

# ==========================================
# FILTER EVENTS BEFORE ECG
# ==========================================

print("\nFiltering events before ECG...")

chartevents = chartevents.merge(
    record_list[["subject_id", "hadm_id", "ecg_time"]],
    on=["subject_id", "hadm_id"],
    how="inner"
)

labevents = labevents.merge(
    record_list[["subject_id", "hadm_id", "ecg_time"]],
    on=["subject_id", "hadm_id"],
    how="inner"
)

# Keep only events BEFORE ECG
chartevents = chartevents[
    chartevents["charttime"] <= chartevents["ecg_time"]
]

labevents = labevents[
    labevents["charttime"] <= labevents["ecg_time"]
]

# ==========================================
# FEATURE EXTRACTION FUNCTION
# ==========================================

def extract_temporal_features(events_df, itemids_dict):

    features = []

    for name, ids in itemids_dict.items():

        subset = events_df[
            (events_df["itemid"].isin(ids)) &
            (events_df["valuenum"].notna())
        ]

        if subset.empty:
            continue

        # Sort by time to get LAST value
        subset = subset.sort_values("charttime")

        last_vals = subset.groupby(
            ["subject_id","hadm_id","study_id"]
        ).tail(1)

        last_vals = last_vals[[
            "subject_id","hadm_id","study_id","valuenum"
        ]].rename(columns={"valuenum": name})

        features.append(last_vals)

    if not features:
        return pd.DataFrame()

    return reduce(
        lambda l, r: pd.merge(
            l, r,
            on=["subject_id","hadm_id","study_id"],
            how="outer"
        ),
        features
    )


# ==========================================
# EXTRACT FEATURES
# ==========================================

print("\nExtracting vital features...")
vital_features = extract_temporal_features(chartevents, vital_itemids)

print("Extracting lab features...")
lab_features = extract_temporal_features(labevents, lab_itemids)

# ==========================================
# MERGE ALL FEATURES
# ==========================================

print("\nMerging all features...")

temporal_features = pd.merge(
    vital_features,
    lab_features,
    on=["subject_id", "hadm_id", "study_id"],
    how="outer"
)

print("Final temporal dataset shape:", temporal_features.shape)

# ==========================================
# SAVE OUTPUT
# ==========================================

print("\nSaving dataset...")

temporal_features.to_parquet(OUTPUT_PATH, index=False)

print("Saved successfully at:", OUTPUT_PATH)