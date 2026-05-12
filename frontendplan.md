# Frontend Implementation Plan: Temporal Multimodal AMI Prediction Platform

## 1. Project Vision
The **Temporal Multimodal AMI Prediction Platform** is not a generic patient upload portal or a black-box machine learning demo. It is a research-grade clinical AI reasoning workstation simulating a real-world cardiology admissions dashboard.

The primary purpose of this frontend is to provide a powerful, highly interpretable interface for physicians and AI researchers to review pre-stored admissions from the MIMIC-IV cohort. The interface focuses deeply on **temporal physiological reasoning** and **interpretability**, visualizing exactly how the model processes multi-timestep 12-lead ECG trajectories, evaluates dynamic troponin evolution, and combines them via hybrid early/late fusion architectures to predict Acute Myocardial Injury (AMI). The UI philosophy emphasizes clinical trust: every prediction must be backed by anatomical, temporal, and clinical evidence.

## 2. Recommended Tech Stack
To deliver a responsive, complex data-visualization-heavy application, we will use the following modern stack:
- **Framework:** Next.js 15 (App Router) for high-performance server-side rendering and routing.
- **UI Library:** React.js
- **Styling:** Tailwind CSS for fast, utility-first UI construction, utilizing a dark-mode clinical theme.
- **State Management:** Zustand for lightweight, fast global state (handling active admissions and viewer states).
- **Data Visualization:** Recharts for clinical trajectory plotting, and Framer Motion for smooth UI transitions.
- **ECG Rendering:** Custom HTML5 Canvas / WebGL implementation or specialized ECG libraries for high-performance 5000-point signal rendering.
- **Backend Integration:** FastAPI for low-latency Python-based ML inference and data serving.
- **Database:** PostgreSQL for robust storage of parsed MIMIC-IV admissions, clinical metadata, and precomputed embeddings.

## 3. System Architecture
The application follows a decoupled, AI-native microservices architecture:
**Frontend (Next.js) → Backend API (FastAPI) → Inference Engine (PyTorch) → Data Layer (PostgreSQL & Memmaps)**

Rather than running heavy, live inference on arbitrary user-uploaded data, the system relies on a curated database of the `refined_temporal_fusion_dataset` (N=40,255). 
- **Stored Test-Set Admissions:** The database holds actual sequential clinical events.
- **Stored Predictions:** Fast retrieval of the 0.7753 F1 Early Fusion model results.
- **Precomputed Attention Maps:** Anatomical and Temporal GRU attention scores are pre-calculated to allow instantaneous loading.
- **Stored ECG Trajectories:** Quick-access retrieval of 4D memmap signals for rendering in the browser.

## 4. Dashboard Architecture
The main dashboard serves as the central hub, designed to look like a high-end ICU/Telemetry monitoring station.
- **Sidebar Navigation:** Links to 'Admissions', 'Model Metrics', 'Case Comparison', and 'Settings'.
- **Top Metrics Bar:** Displays total cohort size, current AMI prevalence, and current model confidence threshold.
- **Temporal Status Indicators:** Visual sparks indicating if a patient has 1, 2, or 3 temporal observations.
- **Admissions Table:** A highly sortable, filterable list of patient records.

| Admission ID | Patient | Max Troponin | ECG Scans | Risk Level | Model Prediction |
|--------------|---------|--------------|-----------|------------|------------------|
| HADM-100234  | P-7742  | 0.45 ng/mL   | 3 (T₀-T₂) | High       | 82% (AMI)        |
| HADM-109441  | P-8821  | 0.02 ng/mL   | 2 (T₀-T₁) | Low        | 12% (Non-AMI)    |

## 5. Admissions Module
Clicking on an admission transforms the UI into a detailed, case-specific workstation.
- **Detailed Patient Profile:** Displays demographics and admission time.
- **Trajectory Metadata:** Shows the exact time deltas between $T_0$, $T_1$, and $T_2$ scans.
- **Comorbidities Box:** Flags critical confounders (e.g., Sepsis, CKD, Heart Failure) that the model must navigate.
- **Overview Summary:** A high-level view of the entire admission's evolution before diving into exact signals.

## 6. Clinical Values Panel
A structured, timeline-based module dedicated to the physiological state of the patient across the admission.
- **Temporal Table:** Columns for $T_0$, $T_1$, and $T_2$ showing:
  - **Troponin (hs-cTnT):** Raw values, Peak Baseline Ratio, and Fold-Rise.
  - **Vitals:** Heart Rate, Blood Pressure (MAP/SBP/DBP).
  - **Labs:** Creatinine, Lactate.
- **Confounder Status:** Boolean indicators for active Sepsis or CKD (which naturally elevate troponin without infarction).
- **Trajectory Plot:** A miniature Recharts line-graph plotting the Troponin velocity curve alongside ECG scan timestamps.

## 7. ECG Visualization Module
The technical centerpiece of the frontend. A highly interactive, browser-based waveform viewer.
- **Synchronized 12-Lead Layout:** Renders the standard 4x3 + Rhythm strip format in high fidelity.
- **Multi-Timestep Comparison:** Tabs or layered opacity views to compare $T_0$, $T_1$, and $T_2$ morphologies directly.
- **Delta Waveform Highlighting:** Automatically shades regions where the $T_1$ waveform deviated significantly from the $T_0$ baseline (e.g., dynamic ST elevation).
- **Controls:** Features Zoom, Pan, specific lead isolation, and a smooth playback/animation slider that moves across the 10-second (5000 point) sequence.

## 8. Anatomical Attention Visualization
This section bridges the gap between raw data and the model's neural network logic by mapping `LeadAwareHybridExtractor` attention weights to human anatomy.
- **Regional Heatmaps:** Leads are grouped into clinical regions:
  - *Inferior* (II, III, aVF)
  - *Anterior/Septal* (V1, V2, V3, V4)
  - *Lateral* (I, aVL, V5, V6)
- **Visual Overlays:** An anatomical heart diagram glows in the specific regions the AI focused on.
- **Temporal Shift:** Shows how the model's attention shifted from the Anterior leads at $T_0$ to the Inferior leads at $T_1$.

## 9. Prediction Module
The actionable inference station located prominently on the dashboard.
- **"Run AI Analysis" Action:** Simulates a live inference request (retrieving stored model outputs).
- **Prediction Output:** Displays the final probability (e.g., 88.4%) against the optimized baseline threshold.
- **Modality Breakdown:** Shows how the model ingested both clinical context and ECG morphology during the Early Fusion phase.
- **Confidence Gauge:** A radial dial showing the model's certainty.
- **Dominant Modality:** Indicates whether the model relied heavier on the ECG morphological trajectory or the clinical troponin trajectory to make its decision.

## 10. Explainability Panel
A critical feature translating dense neural activations into human-readable clinical narratives using LLM/rules-based generation.
- **Narrative Example:** *"The model detected evolving inferior-wall ischemic morphology (attention maxed on Leads II, aVF at $T_1$) alongside a rapidly rising troponin velocity (+0.4 ng/mL/hr), resulting in an 88% AMI probability."*
- **Explainability Breakdown:**
  - Clearly explains the **Temporal Reasoning** (e.g., "The model gated out the CKD confounder because the ECG changed dynamically").
  - Translates the spatial attention vectors into plain English.

## 11. Temporal Reasoning Timeline
A dynamic horizontal timeline summarizing the entire admission.
- **Progression Flow:** Starts at $T_0$ and moves rightward.
- **Event Markers:** Drops pins on the timeline for Troponin draws and ECG scans.
- **Confidence Evolution:** Plots a line showing how the AI's probability of AMI evolved at each timestep (e.g., 20% at $T_0$ -> 85% at $T_1$).

## 12. Compare Cases Module
An educational and analytical tool allowing researchers to put two admissions side-by-side.
- **CKD False Positive vs True AMI:** Compare an admission with chronically elevated troponin against one with true acute ischemia.
- **Attention Analysis:** View how the model correctly ignores noisy baseline wander in one case while heavily attending to clean ST-elevation in another.

## 13. Backend API Design
The FastAPI layer connecting the Next.js frontend to the SQLite/PostgreSQL database and PyTorch models:
- `GET /api/admissions` - Returns paginated, filterable patient lists.
- `GET /api/admission/{hadm_id}` - Fetches full static clinical and trajectory metadata.
- `GET /api/ecg/{hadm_id}?timestep={t}` - Returns the `12x5000` array for browser rendering.
- `GET /api/prediction/{hadm_id}` - Retrieves the Early Fusion model prediction and confidence intervals.
- `GET /api/explain/{hadm_id}` - Returns the pre-calculated anatomical and temporal attention weights (Entropy/Dominance scores) and textual explanation.

## 14. Database Schema
A highly structured relational or NoSQL equivalent database format:
- **`patients`**: `subject_id`, `age`, `gender`.
- **`admissions`**: `hadm_id`, `subject_id`, `admittime`, `ground_truth_ami`.
- **`clinical_timelines`**: `hadm_id`, `timestep`, `trop_value`, `hr`, `bp`, `ckd`, `sepsis`.
- **`ecg_metadata`**: `hadm_id`, `timestep`, `ecg_path`, `time_delta`.
- **`model_inference`**: `hadm_id`, `predicted_prob`, `attn_temp_entropy`, `attn_spatial_dominance`.

## 15. Folder Structure
Organized for scale and maintainability inside a Next.js environment:
```
/frontend
  /app
    /dashboard         # Main admissions table page
    /admission/[id]    # Specific patient workstation
    /compare           # Side-by-side analysis
  /components
    /ui                # Reusable buttons, cards, modals
    /ecg-viewer        # Canvas/WebGL waveform renderer
    /attention         # Anatomical heart overlay components
    /clinical          # Troponin charts and timeline
  /lib
    /store             # Zustand global state
    /api               # Axios/Fetch wrappers for FastAPI
    /utils             # Signal processing & formatting utils
/backend
  /api                 # FastAPI routes
  /core                # PyTorch model loaders
  /database            # SQLModel / SQLAlchemy schemas
```

## 16. UI/UX Theme
The aesthetics are designed to be striking, professional, and visually clear:
- **Background:** Deep slate/black (`bg-gray-900`) for high contrast.
- **ECG Styling:** Classic neon-green or cyan waveforms on a subtle grid background, replicating physical ECG paper but modernized.
- **AI Highlights:** Vibrant amber or magenta glowing overlays to indicate attention heatmaps.
- **Typography:** Modern, readable sans-serif (e.g., Inter, Roboto) for dense clinical data.
- **Animations:** Smooth Framer Motion transitions when switching between temporal timesteps to maintain spatial context.

## 17. MVP Development Roadmap
- **Phase 1: Architecture & Data Layer** - Initialize Next.js + FastAPI. Create database schema and populate with the `refined_temporal_fusion_dataset`.
- **Phase 2: Dashboard MVP** - Build the admissions table, layout shell, and basic clinical data fetching.
- **Phase 3: ECG Visualization** - Implement the high-performance 12-lead canvas renderer and temporal switching logic.
- **Phase 4: AI & Explainability Integration** - Connect the prediction module, render attention heatmaps, and build the temporal confidence timeline.
- **Phase 5: Polish & Deployment** - Add Framer Motion animations, finalize the dark-mode aesthetics, and deploy via Vercel / Docker.

## 18. Future Enhancements
- **Real-Time Streaming:** Connecting the platform to a live HL7/FHIR hospital feed to predict AMI in real-time as lab results arrive.
- **Self-Supervised Integration:** Visualizing the latent space embeddings of the 800k unlabelled MIMIC ECGs to find "similar past patients."
- **Physician Feedback Loop:** Adding a button for cardiologists to flag "Disagree with AI," automatically triggering active learning / hard-negative mining in the PyTorch backend.
- **Generative AI Reports:** Using Gemini/GPT-4 to write full admission discharge summaries based on the Temporal GRU outputs.

## 19. Final Product Vision
The resulting application is **an interpretable temporal physiological reasoning platform for acute myocardial injury analysis.** By moving away from black-box scores and instead visualizing the exact morphological deltas and temporal attention the AI utilizes, this frontend bridges the gap between deep learning complexity and clinical necessity, setting a new standard for AI cardiology tools.
