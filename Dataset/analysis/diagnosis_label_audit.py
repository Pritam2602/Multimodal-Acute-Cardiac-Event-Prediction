import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIAG_PATH = PROJECT_ROOT / "mimic_data" / "diagnoses_icd.csv.gz"
DEFAULT_D_ICD_PATH = PROJECT_ROOT / "mimic_data" / "d_icd_diagnoses.csv.gz"
DEFAULT_OUT_DIR = PROJECT_ROOT / "Dataset" / "analysis" / "outputs" / "diagnosis_label_audit"


def create_acute_mi_flag(diag: pd.DataFrame) -> pd.Series:
    code = (
        diag["icd_code"]
        .astype(str)
        .str.upper()
        .str.replace(".", "", regex=False)
        .str.strip()
    )
    version = diag["icd_version"].astype(int)

    icd9_acute_mi = (
        (version == 9)
        & code.str.startswith("410")
        & code.str.len().ge(5)
        & code.str[4].isin(["1", "2"])
    )
    icd10_acute_mi = (
        (version == 10)
        & (
            code.str.startswith("I21")
            | code.str.startswith("I22")
        )
    )

    return (icd9_acute_mi | icd10_acute_mi).astype(int)


def main():
    parser = argparse.ArgumentParser(description="Audit AMI ICD label definitions")
    parser.add_argument("--diag-path", type=Path, default=DEFAULT_DIAG_PATH)
    parser.add_argument("--d-icd-path", type=Path, default=DEFAULT_D_ICD_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    diag = pd.read_csv(args.diag_path, usecols=["hadm_id", "icd_code", "icd_version"])
    d_icd = pd.read_csv(args.d_icd_path)
    merged = diag.merge(d_icd, on=["icd_code", "icd_version"], how="left")

    merged["broad_text_mi"] = merged["long_title"].str.contains(
        "myocardial infarction",
        case=False,
        na=False,
    ).astype(int)
    merged["acute_code_mi"] = create_acute_mi_flag(merged)

    title_counts = (
        merged[merged["broad_text_mi"].eq(1)]
        .groupby(["icd_version", "icd_code", "long_title"], dropna=False)
        .agg(
            rows=("hadm_id", "size"),
            admissions=("hadm_id", "nunique"),
            acute_code_mi=("acute_code_mi", "max"),
        )
        .reset_index()
        .sort_values("rows", ascending=False)
    )
    title_counts.to_csv(args.out_dir / "myocardial_infarction_title_counts.csv", index=False)

    admission_labels = merged.groupby("hadm_id").agg(
        broad_text_mi=("broad_text_mi", "max"),
        acute_code_mi=("acute_code_mi", "max"),
    )
    confusion = pd.crosstab(
        admission_labels["broad_text_mi"],
        admission_labels["acute_code_mi"],
        rownames=["broad_text_mi"],
        colnames=["acute_code_mi"],
    )
    confusion.to_csv(args.out_dir / "admission_label_crosstab.csv")

    summary = {
        "diagnosis_rows": int(len(merged)),
        "broad_text_mi_rows": int(merged["broad_text_mi"].sum()),
        "acute_code_mi_rows": int(merged["acute_code_mi"].sum()),
        "broad_text_mi_admissions": int(admission_labels["broad_text_mi"].sum()),
        "acute_code_mi_admissions": int(admission_labels["acute_code_mi"].sum()),
        "broad_only_admissions": int(
            ((admission_labels["broad_text_mi"] == 1) & (admission_labels["acute_code_mi"] == 0)).sum()
        ),
        "acute_only_admissions": int(
            ((admission_labels["broad_text_mi"] == 0) & (admission_labels["acute_code_mi"] == 1)).sum()
        ),
    }
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(" DIAGNOSIS LABEL AUDIT")
    print("=" * 70)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nTop broad-text MI titles:")
    print(title_counts.head(20).to_string(index=False))
    print(f"\n[SAVE] Audit -> {args.out_dir}")


if __name__ == "__main__":
    main()
