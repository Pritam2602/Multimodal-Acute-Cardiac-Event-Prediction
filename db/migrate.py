import os
from sqlalchemy import create_engine, text

DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:pritu%402602@localhost:5432/ami_platform")

def migrate():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        print("Migrating schema...")
        commands = [
            "ALTER TABLE admissions ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();",
            "ALTER TABLE admissions ADD COLUMN IF NOT EXISTS risk_level VARCHAR(20) DEFAULT 'Low';",
            "ALTER TABLE admissions ADD COLUMN IF NOT EXISTS prediction JSONB DEFAULT '{}';",
            "ALTER TABLE admissions ADD COLUMN IF NOT EXISTS attention JSONB DEFAULT '{}';",
            "ALTER TABLE admissions ADD COLUMN IF NOT EXISTS comorbidities JSONB DEFAULT '[]';",
            "ALTER TABLE admissions ADD COLUMN IF NOT EXISTS timelines JSONB DEFAULT '[]';",
            "ALTER TABLE admissions ADD COLUMN IF NOT EXISTS max_troponin FLOAT DEFAULT 0;",
            "ALTER TABLE admissions ADD COLUMN IF NOT EXISTS ecg_seq_len INT DEFAULT 1;",
            "ALTER TABLE admissions ADD COLUMN IF NOT EXISTS explanation TEXT DEFAULT '';",
            "ALTER TABLE admissions ADD COLUMN IF NOT EXISTS explanation_temporal TEXT DEFAULT '';"
        ]
        
        for cmd in commands:
            conn.execute(text(cmd))
        
        # Populate basic timelines JSONB from clinical_timelines table
        # We'll just do a quick basic migration for now so it doesn't crash
        conn.commit()
        print("Schema migration successful.")

if __name__ == "__main__":
    migrate()
