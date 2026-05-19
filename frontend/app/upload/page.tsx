"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { Upload, CheckCircle2, Loader2, AlertCircle, User, Activity, Cpu } from "lucide-react";
import Sidebar from "@/components/ui/Sidebar";
import { useAdmissionStore } from "@/lib/store/useAdmissionStore";

interface FormState {
  // Demographics
  anchor_age: string;
  gender: string;
  // Vitals
  heart_rate: string;
  respiratory_rate: string;
  // Troponin
  troponin_t: string;
  // Labs
  creatinine: string;
  sodium: string;
  potassium: string;
  num_diagnoses: string;
  los: string;
  // ECG intervals
  pr_interval: string;
  qrs_duration: string;
  qt_interval: string;
  qtc: string;
  p_axis: string;
  qrs_axis: string;
  t_axis: string;
  rr_interval: string;
  // Comorbidities
  has_ckd: boolean;
  has_hf: boolean;
  has_sepsis: boolean;
  has_pe: boolean;
  has_afib: boolean;
  has_diabetes: boolean;
  // Optional
  ground_truth_ami: boolean;
  notes: string;
}

const DEFAULTS: FormState = {
  anchor_age: "", gender: "M",
  heart_rate: "", respiratory_rate: "",
  troponin_t: "", creatinine: "", sodium: "", potassium: "",
  num_diagnoses: "", los: "",
  pr_interval: "", qrs_duration: "", qt_interval: "", qtc: "",
  p_axis: "", qrs_axis: "", t_axis: "", rr_interval: "",
  has_ckd: false, has_hf: false, has_sepsis: false,
  has_pe: false, has_afib: false, has_diabetes: false,
  ground_truth_ami: false,
  notes: "",
};

function Field({
  label, name, value, unit, placeholder, onChange,
}: {
  label: string; name: string; value: string; unit?: string;
  placeholder?: string; onChange: (n: string, v: string) => void;
}) {
  return (
    <div>
      <label className="block text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-1">
        {label}{unit && <span className="text-text-muted/60 font-normal ml-1">({unit})</span>}
      </label>
      <input
        type="number"
        step="any"
        value={value}
        placeholder={placeholder ?? "—"}
        onChange={e => onChange(name, e.target.value)}
        className="w-full bg-elevated border border-border-default rounded-lg px-3 py-2 text-xs font-mono text-text-primary focus:outline-none focus:border-cyan/50 placeholder:text-text-muted/40"
      />
    </div>
  );
}

function CheckField({
  label, name, value, onChange,
}: {
  label: string; name: string; value: boolean; onChange: (n: string, v: boolean) => void;
}) {
  return (
    <label className="flex items-center gap-2 cursor-pointer group">
      <div
        onClick={() => onChange(name, !value)}
        className={`w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 transition-all ${
          value ? "bg-cyan/20 border-cyan" : "border-border-default"
        }`}
      >
        {value && <div className="w-2 h-2 rounded-sm bg-cyan" />}
      </div>
      <span className="text-[11px] text-text-secondary group-hover:text-text-primary transition-colors">{label}</span>
    </label>
  );
}

export default function UploadPage() {
  const router = useRouter();
  const { fetchAdmissions } = useAdmissionStore();
  const [form, setForm] = useState<FormState>(DEFAULTS);
  const [status, setStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [errorMsg, setErrorMsg] = useState("");
  const [newId, setNewId] = useState("");

  const setStr = (name: string, val: string) => setForm(f => ({ ...f, [name]: val }));
  const setBool = (name: string, val: boolean) => setForm(f => ({ ...f, [name]: val }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setStatus("loading");
    setErrorMsg("");

    const payload = {
      anchor_age:       parseFloat(form.anchor_age) || undefined,
      gender:           form.gender,
      heart_rate:       parseFloat(form.heart_rate) || undefined,
      respiratory_rate: parseFloat(form.respiratory_rate) || undefined,
      troponin_t:       parseFloat(form.troponin_t) || undefined,
      creatinine:       parseFloat(form.creatinine) || undefined,
      sodium:           parseFloat(form.sodium) || undefined,
      potassium:        parseFloat(form.potassium) || undefined,
      num_diagnoses:    parseFloat(form.num_diagnoses) || undefined,
      los:              parseFloat(form.los) || undefined,
      pr_interval:      parseFloat(form.pr_interval) || undefined,
      qrs_duration:     parseFloat(form.qrs_duration) || undefined,
      qt_interval:      parseFloat(form.qt_interval) || undefined,
      qtc:              parseFloat(form.qtc) || undefined,
      p_axis:           parseFloat(form.p_axis) || undefined,
      qrs_axis:         parseFloat(form.qrs_axis) || undefined,
      t_axis:           parseFloat(form.t_axis) || undefined,
      rr_interval:      parseFloat(form.rr_interval) || undefined,
      has_ckd:          form.has_ckd,
      has_hf:           form.has_hf,
      has_sepsis:       form.has_sepsis,
      has_pe:           form.has_pe,
      has_afib:         form.has_afib,
      has_diabetes:     form.has_diabetes,
      ground_truth_ami: form.ground_truth_ami,
      notes:            form.notes,
    };

    try {
      const res = await fetch("/api/admissions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || data.error || "Upload failed");
      setNewId(data.admission.hadm_id);
      setStatus("success");
      // Refresh store so dashboard count and list update immediately
      fetchAdmissions();
    } catch (err) {
      setErrorMsg(String(err));
      setStatus("error");
    }
  };

  if (status === "success") {
    return (
      <div className="flex h-screen overflow-hidden bg-base">
        <Sidebar />
        <main className="flex-1 flex flex-col items-center justify-center gap-6">
          <div className="flex flex-col items-center gap-4 text-center max-w-sm">
            <div className="w-16 h-16 rounded-full bg-safe/10 border border-safe/30 flex items-center justify-center">
              <CheckCircle2 className="w-8 h-8 text-safe" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-text-primary mb-1">Patient Uploaded</h2>
              <p className="text-xs text-text-muted">AI analysis complete. Admission record created.</p>
              <p className="text-xs font-mono text-cyan mt-2">{newId}</p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={() => router.push(`/admission/${encodeURIComponent(newId)}`)}
                className="px-4 py-2 text-xs font-semibold bg-cyan/10 border border-cyan/40 text-cyan rounded-lg hover:bg-cyan/15"
              >
                View Admission
              </button>
              <button
                onClick={() => router.push("/dashboard")}
                className="px-4 py-2 text-xs font-semibold bg-elevated border border-border-default text-text-secondary rounded-lg hover:text-text-primary"
              >
                Back to Dashboard
              </button>
              <button
                onClick={() => { setForm(DEFAULTS); setStatus("idle"); }}
                className="px-4 py-2 text-xs font-semibold bg-elevated border border-border-default text-text-secondary rounded-lg hover:text-text-primary"
              >
                Upload Another
              </button>
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-base">
      <Sidebar />
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-6 py-8">
          {/* Header */}
          <div className="flex items-center gap-3 mb-8">
            <div className="w-8 h-8 rounded-lg bg-cyan/10 border border-cyan/30 flex items-center justify-center">
              <Upload className="w-4 h-4 text-cyan" />
            </div>
            <div>
              <h1 className="text-base font-bold text-text-primary">Upload Patient Admission</h1>
              <p className="text-[11px] text-text-muted">Enter clinical values — AI analysis runs automatically on submission</p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Demographics */}
            <div className="card p-5">
              <div className="flex items-center gap-2 mb-4">
                <User className="w-4 h-4 text-cyan" />
                <h2 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Demographics</h2>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <Field label="Age" name="anchor_age" value={form.anchor_age} unit="years" placeholder="e.g. 65" onChange={setStr} />
                <div>
                  <label className="block text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-1">Gender</label>
                  <select
                    value={form.gender}
                    onChange={e => setStr("gender", e.target.value)}
                    className="w-full bg-elevated border border-border-default rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-cyan/50"
                  >
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Vitals & Labs */}
            <div className="card p-5">
              <div className="flex items-center gap-2 mb-4">
                <Activity className="w-4 h-4 text-danger" />
                <h2 className="text-xs font-semibold text-text-primary uppercase tracking-wider">Vitals & Lab Values</h2>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <Field label="Heart Rate" name="heart_rate" value={form.heart_rate} unit="bpm" placeholder="75" onChange={setStr} />
                <Field label="Respiratory Rate" name="respiratory_rate" value={form.respiratory_rate} unit="breaths/min" placeholder="16" onChange={setStr} />
                <Field label="Troponin T" name="troponin_t" value={form.troponin_t} unit="ng/mL" placeholder="0.01" onChange={setStr} />
                <Field label="Creatinine" name="creatinine" value={form.creatinine} unit="mg/dL" placeholder="1.0" onChange={setStr} />
                <Field label="Sodium" name="sodium" value={form.sodium} unit="mEq/L" placeholder="138" onChange={setStr} />
                <Field label="Potassium" name="potassium" value={form.potassium} unit="mEq/L" placeholder="4.0" onChange={setStr} />
                <Field label="# Diagnoses" name="num_diagnoses" value={form.num_diagnoses} placeholder="5" onChange={setStr} />
                <Field label="Length of Stay" name="los" value={form.los} unit="days" placeholder="2.5" onChange={setStr} />
              </div>
            </div>

            {/* ECG Intervals */}
            <div className="card p-5">
              <div className="flex items-center gap-2 mb-4">
                <Cpu className="w-4 h-4 text-purple" />
                <h2 className="text-xs font-semibold text-text-primary uppercase tracking-wider">ECG Intervals</h2>
              </div>
              <div className="grid grid-cols-4 gap-4">
                <Field label="PR Interval" name="pr_interval" value={form.pr_interval} unit="ms" placeholder="160" onChange={setStr} />
                <Field label="QRS Duration" name="qrs_duration" value={form.qrs_duration} unit="ms" placeholder="90" onChange={setStr} />
                <Field label="QT Interval" name="qt_interval" value={form.qt_interval} unit="ms" placeholder="380" onChange={setStr} />
                <Field label="QTc" name="qtc" value={form.qtc} unit="ms" placeholder="440" onChange={setStr} />
                <Field label="P Axis" name="p_axis" value={form.p_axis} unit="°" placeholder="60" onChange={setStr} />
                <Field label="QRS Axis" name="qrs_axis" value={form.qrs_axis} unit="°" placeholder="45" onChange={setStr} />
                <Field label="T Axis" name="t_axis" value={form.t_axis} unit="°" placeholder="30" onChange={setStr} />
                <Field label="RR Interval" name="rr_interval" value={form.rr_interval} unit="ms" placeholder="800" onChange={setStr} />
              </div>
            </div>

            {/* Comorbidities */}
            <div className="card p-5">
              <h2 className="text-xs font-semibold text-text-primary uppercase tracking-wider mb-4">Comorbidities</h2>
              <div className="grid grid-cols-3 gap-3">
                <CheckField label="Chronic Kidney Disease (CKD)" name="has_ckd" value={form.has_ckd} onChange={setBool} />
                <CheckField label="Heart Failure" name="has_hf" value={form.has_hf} onChange={setBool} />
                <CheckField label="Sepsis" name="has_sepsis" value={form.has_sepsis} onChange={setBool} />
                <CheckField label="Pulmonary Embolism" name="has_pe" value={form.has_pe} onChange={setBool} />
                <CheckField label="Atrial Fibrillation" name="has_afib" value={form.has_afib} onChange={setBool} />
                <CheckField label="Diabetes" name="has_diabetes" value={form.has_diabetes} onChange={setBool} />
              </div>
            </div>

            {/* Notes */}
            <div className="card p-5">
              <label className="block text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-2">
                Clinical Notes (optional)
              </label>
              <textarea
                value={form.notes}
                onChange={e => setStr("notes", e.target.value)}
                placeholder="Any additional clinical context..."
                rows={3}
                className="w-full bg-elevated border border-border-default rounded-lg px-3 py-2 text-xs text-text-primary focus:outline-none focus:border-cyan/50 placeholder:text-text-muted/40 resize-none"
              />
            </div>

            {status === "error" && (
              <div className="flex items-center gap-2 p-3 rounded-lg bg-danger/5 border border-danger/20 text-danger text-xs">
                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                {errorMsg}
              </div>
            )}

            {/* Submit */}
            <div className="flex justify-end gap-3 pb-8">
              <button
                type="button"
                onClick={() => router.push("/dashboard")}
                className="px-5 py-2.5 text-xs font-semibold bg-elevated border border-border-default text-text-secondary rounded-lg hover:text-text-primary"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={status === "loading"}
                className="flex items-center gap-2 px-5 py-2.5 text-xs font-semibold bg-cyan/10 border border-cyan/40 text-cyan rounded-lg hover:bg-cyan/15 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {status === "loading" ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Running AI Analysis...</>
                ) : (
                  <><Upload className="w-4 h-4" /> Upload & Analyse</>
                )}
              </button>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}
