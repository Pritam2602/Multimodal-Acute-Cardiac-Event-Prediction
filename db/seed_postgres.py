import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

# Connection string (adjust password if yours is not 'postgres')
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:pritu%402602@localhost:5432/ami_platform")

def clean_value(v):
    # Pandas NaN to Python None for SQL insertion
    if pd.isna(v):
        return None
    return float(v) if isinstance(v, (np.float32, np.float64)) else v

def seed_database():
    print(f"Connecting to database: {DB_URL}")
    engine = create_engine(DB_URL)
    
    # 1. Create Tables
    with engine.connect() as conn:
        print("Creating tables...")
        conn.execute(text("""
            DROP TABLE IF EXISTS model_inference CASCADE;
            DROP TABLE IF EXISTS clinical_timelines CASCADE;
            DROP TABLE IF EXISTS ecg_metadata CASCADE;
            DROP TABLE IF EXISTS admissions CASCADE;
            DROP TABLE IF EXISTS patients CASCADE;

            CREATE TABLE patients (
                subject_id VARCHAR(50) PRIMARY KEY,
                age FLOAT,
                gender INT
            );

            CREATE TABLE admissions (
                hadm_id VARCHAR(50) PRIMARY KEY,
                subject_id VARCHAR(50) REFERENCES patients(subject_id),
                ground_truth_ami INT,
                has_ckd INT,
                has_sepsis INT,
                has_hf INT
            );

            CREATE TABLE clinical_timelines (
                id SERIAL PRIMARY KEY,
                hadm_id VARCHAR(50) REFERENCES admissions(hadm_id),
                timestep INT,
                trop_value FLOAT,
                trop_time FLOAT,
                heart_rate FLOAT,
                respiratory_rate FLOAT,
                creatinine FLOAT
            );

            CREATE TABLE ecg_metadata (
                id SERIAL PRIMARY KEY,
                hadm_id VARCHAR(50) REFERENCES admissions(hadm_id),
                timestep INT,
                ecg_path VARCHAR(255),
                ecg_time TIMESTAMP
            );
        """))
        conn.commit()

    # 2. Load Data
    print("Loading refined dataset...")
    df = pd.read_parquet("D:/MINI_PROJECT/mimic_data/refined_temporal_fusion_dataset.parquet")
    
    # The dataset uses 'hadm_id' primarily since 'subject_id' was missing in some early cuts, 
    # but we can synthesize a subject_id = "PAT_" + hadm_id if it's missing
    if 'subject_id' not in df.columns:
        df['subject_id'] = "PAT_" + df['hadm_id'].astype(str)

    # 3. Insert Patients
    print("Seeding patients...")
    patients_df = df[['subject_id', 'anchor_age', 'gender']].drop_duplicates('subject_id')
    patients_df.rename(columns={'anchor_age': 'age'}, inplace=True)
    patients_df.to_sql('patients', engine, if_exists='append', index=False)

    # 4. Insert Admissions
    print("Seeding admissions...")
    admissions_df = df[['hadm_id', 'subject_id', 'AMI', 'has_ckd', 'has_sepsis', 'has_hf']].copy()
    admissions_df.rename(columns={'AMI': 'ground_truth_ami'}, inplace=True)
    admissions_df.to_sql('admissions', engine, if_exists='append', index=False)

    # 5. Insert Clinical Timelines & ECG Metadata
    print("Seeding timelines & ECGs (this may take a minute)...")
    clinical_rows = []
    ecg_rows = []
    
    for _, row in df.iterrows():
        hadm = row['hadm_id']
        hr = clean_value(row['Heart_Rate'])
        rr = clean_value(row['Respiratory_Rate'])
        creat = clean_value(row['Creatinine'])
        
        for t in range(3):
            # Trop
            trop_val = clean_value(row.get(f'trop_{t}'))
            trop_time = clean_value(row.get(f'trop_{t}_time'))
            if trop_val is not None:
                clinical_rows.append({
                    'hadm_id': hadm, 'timestep': t,
                    'trop_value': trop_val, 'trop_time': trop_time,
                    'heart_rate': hr, 'respiratory_rate': rr, 'creatinine': creat
                })
                
            # ECG
            ecg_path = row.get(f'ecg_path_{t}')
            ecg_time = clean_value(row.get(f'ecg_time_{t}'))
            if pd.notna(ecg_path):
                ecg_rows.append({
                    'hadm_id': hadm, 'timestep': t,
                    'ecg_path': str(ecg_path), 'ecg_time': ecg_time
                })

    pd.DataFrame(clinical_rows).to_sql('clinical_timelines', engine, if_exists='append', index=False)
    pd.DataFrame(ecg_rows).to_sql('ecg_metadata', engine, if_exists='append', index=False)

    print("✅ Database seeding complete!")

if __name__ == "__main__":
    seed_database()
