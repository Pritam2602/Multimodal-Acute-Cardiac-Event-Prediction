-- ============================================================
-- Temporal Multimodal AMI Prediction Platform — PostgreSQL Schema
-- ============================================================
-- Run this once to set up the database, then import test data:
--   psql -U postgres -d ami_platform -f db/schema.sql
--
-- Import test split (after running split_dataset.py):
--   \COPY admissions FROM 'splits/test_for_postgres.csv' CSV HEADER;
-- ============================================================

-- Drop if re-creating (comment out for production)
DROP TABLE IF EXISTS admissions CASCADE;
DROP TABLE IF EXISTS upload_log CASCADE;

-- ── Main admissions table ────────────────────────────────────
CREATE TABLE admissions (
    -- Identity
    hadm_id             BIGINT       PRIMARY KEY,
    subject_id          TEXT         NOT NULL DEFAULT '',
    anchor_age          FLOAT,
    gender              CHAR(1)      CHECK (gender IN ('M', 'F')),
    admittime           TIMESTAMPTZ  DEFAULT NOW(),

    -- Vital signs / Clinical
    heart_rate          FLOAT,
    respiratory_rate    FLOAT,
    troponin_t          FLOAT,
    creatinine          FLOAT,
    sodium              FLOAT,
    potassium           FLOAT,
    num_diagnoses       FLOAT,
    los                 FLOAT,

    -- ECG intervals
    pr_interval         FLOAT,
    qrs_duration        FLOAT,
    qt_interval         FLOAT,
    qtc                 FLOAT,
    p_axis              FLOAT,
    qrs_axis            FLOAT,
    t_axis              FLOAT,
    rr_interval         FLOAT,

    -- Troponin trajectory summary
    troponin_peak       FLOAT,
    troponin_velocity   FLOAT,
    trop_fold_rise      FLOAT,
    max_troponin        FLOAT,
    ecg_seq_len         INTEGER      DEFAULT 1,

    -- Comorbidity flags
    has_ckd             BOOLEAN      DEFAULT FALSE,
    has_hf              BOOLEAN      DEFAULT FALSE,
    has_sepsis          BOOLEAN      DEFAULT FALSE,
    has_pe              BOOLEAN      DEFAULT FALSE,
    has_afib            BOOLEAN      DEFAULT FALSE,
    has_diabetes        BOOLEAN      DEFAULT FALSE,

    -- Ground truth & risk
    ground_truth_ami    BOOLEAN      NOT NULL DEFAULT FALSE,
    risk_level          TEXT         CHECK (risk_level IN ('Low', 'Moderate', 'High', 'Critical')),

    -- Complex nested data stored as JSONB
    -- timelines: ClinicalTimestep[]
    timelines           JSONB        NOT NULL DEFAULT '[]',
    -- prediction: PredictionData
    prediction          JSONB        NOT NULL DEFAULT '{}',
    -- attention: AttentionWeights
    attention           JSONB        NOT NULL DEFAULT '{}',
    -- comorbidities: string[]
    comorbidities       JSONB        NOT NULL DEFAULT '[]',

    -- AI narrative
    explanation         TEXT         DEFAULT '',
    explanation_temporal TEXT        DEFAULT '',

    -- Metadata
    split               TEXT         DEFAULT 'test',   -- 'train' | 'val' | 'test' | 'live'
    created_at          TIMESTAMPTZ  DEFAULT NOW(),
    updated_at          TIMESTAMPTZ  DEFAULT NOW()
);

-- ── Indexes for fast frontend queries ───────────────────────
CREATE INDEX idx_admissions_risk       ON admissions (risk_level);
CREATE INDEX idx_admissions_ami        ON admissions (ground_truth_ami);
CREATE INDEX idx_admissions_split      ON admissions (split);
CREATE INDEX idx_admissions_created    ON admissions (created_at DESC);
CREATE INDEX idx_admissions_pred_prob  ON admissions ((((prediction->>'predicted_prob')::float)));

-- ── Doctor upload log (audit trail) ─────────────────────────
CREATE TABLE upload_log (
    id          SERIAL       PRIMARY KEY,
    hadm_id     BIGINT       REFERENCES admissions (hadm_id) ON DELETE CASCADE,
    uploaded_by TEXT         DEFAULT 'doctor',
    uploaded_at TIMESTAMPTZ  DEFAULT NOW(),
    notes       TEXT
);

-- ── Helper view for the frontend admissions list ─────────────
CREATE OR REPLACE VIEW v_admissions_list AS
SELECT
    hadm_id,
    subject_id,
    anchor_age                                          AS age,
    gender,
    admittime,
    ground_truth_ami,
    max_troponin,
    ecg_seq_len                                         AS ecg_count,
    risk_level,
    (prediction->>'predicted_prob')::float              AS predicted_prob,
    (prediction->>'threshold')::float                   AS threshold,
    comorbidities,
    split,
    created_at
FROM admissions
ORDER BY created_at DESC;

-- ── Auto-update updated_at trigger ──────────────────────────
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_admissions_updated
BEFORE UPDATE ON admissions
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ── Verify schema ────────────────────────────────────────────
SELECT
    'Schema created successfully.' AS status,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'admissions') AS admissions_columns,
    (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'upload_log') AS upload_log_columns;
