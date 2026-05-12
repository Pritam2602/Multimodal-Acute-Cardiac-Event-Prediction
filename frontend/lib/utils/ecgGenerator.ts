"use client";

// Generates realistic synthetic 12-lead ECG waveforms for display
// Based on mathematical models of P/QRS/T morphology

export const LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];

const SAMPLE_RATE = 500; // Hz
const DURATION = 10;    // seconds
const N_SAMPLES = SAMPLE_RATE * DURATION; // 5000

function gaussian(x: number, mu: number, sigma: number): number {
  return Math.exp(-Math.pow(x - mu, 2) / (2 * sigma * sigma));
}

function generatePQRST(
  t: number,
  heartRate: number,
  pAmp: number,
  qAmp: number,
  rAmp: number,
  sAmp: number,
  tAmp: number,
  stLevel: number,
  qrsWidth: number,
  invert: boolean
): number {
  const rr = 60 / heartRate;
  const tmod = ((t % rr) + rr) % rr;

  const sign = invert ? -1 : 1;

  // P wave at ~15% of RR
  const pCenter = rr * 0.15;
  const p = sign * pAmp * gaussian(tmod, pCenter, 0.025);

  // Q dip at ~30% of RR
  const qCenter = rr * 0.30;
  const q = -sign * qAmp * gaussian(tmod, qCenter, qrsWidth * 0.3);

  // R peak at ~32% of RR
  const rCenter = rr * 0.32;
  const r = sign * rAmp * gaussian(tmod, rCenter, qrsWidth * 0.5);

  // S dip at ~34% of RR
  const sCenter = rr * 0.34;
  const s = -sign * sAmp * gaussian(tmod, sCenter, qrsWidth * 0.3);

  // ST segment + T wave at ~50% of RR
  const tCenter = rr * 0.50;
  const stSegment = stLevel * (gaussian(tmod, rr * 0.38, 0.03) + gaussian(tmod, rr * 0.42, 0.03));
  const tw = sign * tAmp * gaussian(tmod, tCenter, 0.06);

  return p + q + r + s + stSegment + tw;
}

interface LeadParams {
  pAmp: number;
  qAmp: number;
  rAmp: number;
  sAmp: number;
  tAmp: number;
  stLevel: number;
  qrsWidth: number;
  invert: boolean;
}

// Baseline parameters per lead (normal morphology)
const NORMAL_LEAD_PARAMS: Record<string, LeadParams> = {
  I:   { pAmp: 0.12, qAmp: 0.05, rAmp: 0.80, sAmp: 0.10, tAmp: 0.30, stLevel: 0.00, qrsWidth: 0.018, invert: false },
  II:  { pAmp: 0.18, qAmp: 0.04, rAmp: 1.00, sAmp: 0.08, tAmp: 0.40, stLevel: 0.00, qrsWidth: 0.018, invert: false },
  III: { pAmp: 0.08, qAmp: 0.06, rAmp: 0.60, sAmp: 0.14, tAmp: 0.25, stLevel: 0.00, qrsWidth: 0.018, invert: false },
  aVR: { pAmp: 0.10, qAmp: 0.04, rAmp: 0.30, sAmp: 0.20, tAmp: 0.20, stLevel: 0.00, qrsWidth: 0.018, invert: true  },
  aVL: { pAmp: 0.06, qAmp: 0.08, rAmp: 0.50, sAmp: 0.12, tAmp: 0.18, stLevel: 0.00, qrsWidth: 0.018, invert: false },
  aVF: { pAmp: 0.14, qAmp: 0.05, rAmp: 0.70, sAmp: 0.10, tAmp: 0.28, stLevel: 0.00, qrsWidth: 0.018, invert: false },
  V1:  { pAmp: 0.06, qAmp: 0.04, rAmp: 0.20, sAmp: 0.90, tAmp: -0.15, stLevel: 0.00, qrsWidth: 0.020, invert: false },
  V2:  { pAmp: 0.08, qAmp: 0.03, rAmp: 0.45, sAmp: 0.70, tAmp: 0.25, stLevel: 0.00, qrsWidth: 0.020, invert: false },
  V3:  { pAmp: 0.10, qAmp: 0.03, rAmp: 0.70, sAmp: 0.50, tAmp: 0.35, stLevel: 0.00, qrsWidth: 0.020, invert: false },
  V4:  { pAmp: 0.12, qAmp: 0.04, rAmp: 1.10, sAmp: 0.30, tAmp: 0.45, stLevel: 0.00, qrsWidth: 0.020, invert: false },
  V5:  { pAmp: 0.12, qAmp: 0.05, rAmp: 1.20, sAmp: 0.20, tAmp: 0.40, stLevel: 0.00, qrsWidth: 0.018, invert: false },
  V6:  { pAmp: 0.12, qAmp: 0.06, rAmp: 1.00, sAmp: 0.15, tAmp: 0.35, stLevel: 0.00, qrsWidth: 0.018, invert: false },
};

// Inferior STEMI modifications (II, III, aVF elevation; aVL reciprocal)
function inferiorSTEMIParams(timestep: number): Partial<Record<string, Partial<LeadParams>>> {
  const severity = Math.min(timestep * 0.35, 0.8);
  return {
    II:  { stLevel: 0.25 * severity, tAmp: 0.55, rAmp: 0.85 },
    III: { stLevel: 0.30 * severity, tAmp: 0.50, rAmp: 0.70 },
    aVF: { stLevel: 0.28 * severity, tAmp: 0.50, rAmp: 0.70 },
    aVL: { stLevel: -0.20 * severity },           // reciprocal
    I:   { stLevel: -0.10 * severity },           // reciprocal
  };
}

// Anterior STEMI modifications (V1-V4 elevation)
function anteriorSTEMIParams(timestep: number): Partial<Record<string, Partial<LeadParams>>> {
  const severity = Math.min(timestep * 0.35, 0.9);
  return {
    V1: { stLevel: 0.20 * severity, tAmp: 0.35, rAmp: 0.30 },
    V2: { stLevel: 0.40 * severity, tAmp: 0.60, rAmp: 0.55 },
    V3: { stLevel: 0.45 * severity, tAmp: 0.65, rAmp: 0.80 },
    V4: { stLevel: 0.35 * severity, tAmp: 0.55, rAmp: 0.95 },
    II: { stLevel: -0.10 * severity },
    III: { stLevel: -0.08 * severity },
  };
}

// Lateral STEMI modifications (I, aVL, V5, V6 elevation)
function lateralSTEMIParams(timestep: number): Partial<Record<string, Partial<LeadParams>>> {
  const severity = Math.min(timestep * 0.35, 0.85);
  return {
    I:   { stLevel: 0.22 * severity, tAmp: 0.45 },
    aVL: { stLevel: 0.28 * severity, tAmp: 0.42 },
    V5:  { stLevel: 0.35 * severity, tAmp: 0.55 },
    V6:  { stLevel: 0.32 * severity, tAmp: 0.50 },
    II:  { stLevel: -0.12 * severity },
    III: { stLevel: -0.10 * severity },
    aVF: { stLevel: -0.08 * severity },
  };
}

export type ECGPattern = "normal" | "inferior_stemi" | "anterior_stemi" | "lateral_stemi";

function addNoise(val: number, noiseLevel = 0.012): number {
  return val + (Math.random() - 0.5) * 2 * noiseLevel;
}

export function generateECGSignal(
  lead: string,
  heartRate = 80,
  pattern: ECGPattern = "normal",
  timestep = 0
): Float32Array {
  const params = { ...NORMAL_LEAD_PARAMS[lead] };

  // Apply pattern modifications
  let mods: Partial<Record<string, Partial<LeadParams>>> = {};
  if (pattern === "inferior_stemi") mods = inferiorSTEMIParams(timestep);
  else if (pattern === "anterior_stemi") mods = anteriorSTEMIParams(timestep);
  else if (pattern === "lateral_stemi") mods = lateralSTEMIParams(timestep);

  if (mods[lead]) {
    Object.assign(params, mods[lead]);
  }

  const signal = new Float32Array(N_SAMPLES);
  for (let i = 0; i < N_SAMPLES; i++) {
    const t = i / SAMPLE_RATE;
    const raw = generatePQRST(
      t, heartRate,
      params.pAmp, params.qAmp, params.rAmp, params.sAmp, params.tAmp,
      params.stLevel, params.qrsWidth, params.invert
    );
    signal[i] = addNoise(raw);
  }
  return signal;
}

export function generateAll12Leads(
  heartRate = 80,
  pattern: ECGPattern = "normal",
  timestep = 0
): Record<string, Float32Array> {
  return Object.fromEntries(
    LEAD_NAMES.map(lead => [lead, generateECGSignal(lead, heartRate, pattern, timestep)])
  );
}

// Downsample signal for efficient canvas rendering
export function downsample(signal: Float32Array, targetPoints: number): number[] {
  const ratio = signal.length / targetPoints;
  const result: number[] = new Array(targetPoints);
  for (let i = 0; i < targetPoints; i++) {
    const start = Math.floor(i * ratio);
    const end = Math.floor((i + 1) * ratio);
    let max = -Infinity, min = Infinity;
    for (let j = start; j < end; j++) {
      if (signal[j] > max) max = signal[j];
      if (signal[j] < min) min = signal[j];
    }
    result[i] = (max + min) / 2;
  }
  return result;
}

export function getPatternForAdmission(hadmId: string, timestep: number): { pattern: ECGPattern; hr: number } {
  // Map admission IDs to ECG patterns for consistent rendering
  const patterns: Record<string, ECGPattern> = {
    "HADM-100234": "inferior_stemi",
    "HADM-109441": "normal",
    "HADM-112987": "anterior_stemi",
    "HADM-118203": "normal",
    "HADM-121456": "lateral_stemi",
    "HADM-124788": "normal",
    "HADM-128001": "anterior_stemi",
    "HADM-131244": "normal",
    "HADM-135677": "inferior_stemi",
    "HADM-139022": "normal",
    "HADM-142387": "anterior_stemi",
    "HADM-147890": "normal",
    "HADM-153201": "lateral_stemi",
    "HADM-158432": "normal",
    "HADM-162877": "inferior_stemi",
  };
  const hrMap: Record<string, number[]> = {
    "HADM-100234": [98, 104, 112],
    "HADM-112987": [110, 118, 124],
    "HADM-128001": [95, 102, 108],
    "HADM-142387": [118, 124, 130],
  };
  const hrs = hrMap[hadmId] ?? [75, 80, 82];
  return {
    pattern: patterns[hadmId] ?? "normal",
    hr: hrs[Math.min(timestep, hrs.length - 1)] ?? 75,
  };
}
