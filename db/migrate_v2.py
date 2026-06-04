import os
import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:pritu%402602@localhost:5432/ami_platform")

def migrate():
    engine = create_engine(DB_URL)
    df = pd.read_parquet(r"D:\MINI_PROJECT\mimic_data\refined_temporal_fusion_dataset.parquet")
    
    with engine.connect() as conn:
        # Add missing columns
        print("Adding missing columns to admissions...")
        for col_sql in [
            "ALTER TABLE admissions ADD COLUMN IF NOT EXISTS anchor_age FLOAT DEFAULT 0;",
            "ALTER TABLE admissions ADD COLUMN IF NOT EXISTS gender VARCHAR(2) DEFAULT 'M';",
            "ALTER TABLE admissions ADD COLUMN IF NOT EXISTS admittime TIMESTAMP DEFAULT NOW();",
        ]:
            conn.execute(text(col_sql))
        conn.commit()
        
        # Update age and gender from patients table
        print("Populating age & gender from patients table...")
        conn.execute(text("""
            UPDATE admissions a
            SET anchor_age = COALESCE(p.age, 0),
                gender = CASE WHEN p.gender = 1 THEN 'M' ELSE 'F' END
            FROM patients p
            WHERE a.subject_id = p.subject_id
        """))
        conn.commit()
        
        # Also populate timelines JSONB from clinical_timelines table
        print("Building timelines JSONB from clinical_timelines...")
        rows = conn.execute(text("""
            SELECT hadm_id, 
                   json_agg(
                     json_build_object(
                       'timestep', timestep,
                       'label', 'T' || timestep,
                       'time_delta_hrs', timestep * 3.0,
                       'trop_value', COALESCE(trop_value, 0),
                       'peak_baseline_ratio', 1.0,
                       'fold_rise', 1.0,
                       'hr', COALESCE(heart_rate, 75),
                       'sbp', 120,
                       'dbp', 80,
                       'map', 93,
                       'lactate', 1.2,
                       'creatinine', COALESCE(creatinine, 1.0),
                       'ckd_active', false,
                       'sepsis_active', false
                     ) ORDER BY timestep
                   ) as tl
            FROM clinical_timelines
            GROUP BY hadm_id
        """)).fetchall()
        
        batch = []
        for r in rows:
            import json
            batch.append({"tl": json.dumps(r.tl), "hadm": r.hadm_id})
            if len(batch) >= 5000:
                conn.execute(text("UPDATE admissions SET timelines = CAST(:tl AS jsonb) WHERE hadm_id = :hadm"), batch)
                batch = []
        if batch:
            conn.execute(text("UPDATE admissions SET timelines = CAST(:tl AS jsonb) WHERE hadm_id = :hadm"), batch)
        conn.commit()
        
        # Populate comorbidities JSONB
        print("Building comorbidities JSONB...")
        conn.execute(text("""
            UPDATE admissions
            SET comorbidities = CAST(
                (SELECT COALESCE(CAST(json_agg(c) AS text), '[]')
                FROM (
                    SELECT unnest(ARRAY[
                        CASE WHEN has_ckd = 1 THEN 'CKD' END,
                        CASE WHEN has_sepsis = 1 THEN 'Sepsis' END,
                        CASE WHEN has_hf = 1 THEN 'Heart Failure' END
                    ]) as c
                ) sub
                WHERE c IS NOT NULL)
            AS jsonb)
        """))
        conn.commit()
        
        # Verify
        sample = conn.execute(text("SELECT hadm_id, anchor_age, gender, subject_id FROM admissions LIMIT 5")).fetchall()
        for s in sample:
            print(f"  {s}")
        
        ami = conn.execute(text("SELECT COUNT(*) FROM admissions WHERE ground_truth_ami=1")).scalar()
        total = conn.execute(text("SELECT COUNT(*) FROM admissions")).scalar()
        print(f"\nAMI prevalence: {ami}/{total} = {ami/total*100:.2f}%")
        
    print("Migration complete!")

if __name__ == "__main__":
    migrate()
