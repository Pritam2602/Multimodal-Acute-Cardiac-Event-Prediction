import os
from sqlalchemy import create_engine, text

DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:pritu%402602@localhost:5432/ami_platform")
engine = create_engine(DB_URL)

with engine.connect() as conn:
    total = conn.execute(text("SELECT COUNT(*) FROM admissions")).scalar()
    ami = conn.execute(text("SELECT COUNT(*) FROM admissions WHERE ground_truth_ami = 1")).scalar()
    print(f"Total: {total}")
    print(f"AMI=1: {ami}")
    print(f"Prevalence: {ami/total*100:.2f}%")
    
    # Check patient details
    rows = conn.execute(text("SELECT hadm_id, subject_id, ground_truth_ami, age, gender FROM admissions LIMIT 5")).fetchall()
    for r in rows:
        print(f"  {r}")
    
    # Check what columns exist
    cols = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name='admissions' ORDER BY ordinal_position")).fetchall()
    print(f"\nAdmissions columns: {[c[0] for c in cols]}")
