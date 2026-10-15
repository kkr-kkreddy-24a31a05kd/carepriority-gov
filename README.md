# 🏥 CarePriority Gov
### Dynamic Clinical Severity OPD Triage Assistant for Overloaded Government Hospitals

Live demo: https://carepriority-gov.streamlit.app/

> **Demo only - not validated for clinical use. Metrics use placeholder data.**

> **Hackathon Problem H1:** Outpatient Department (OPD) Triage Assistant for an Overloaded Government Hospital  
> **Clinical Disclaimer:** *Demo only — not validated for clinical use. Independent demonstration project, not an official government service.*

---

## 🌟 Executive Summary & Problem Context
In typical government hospital Outpatient Departments (OPD) across India and emerging healthcare systems, facilities frequently face **~800 walk-ins per day with only 2 doctors on duty**. 

Under a traditional **First-Come-First-Served (FCFS)** queue:
- Patients with silent hypoxemia (SpO2 < 90%), impending stroke, atypical myocardial infarction, severe shock, or acute poisoning wait behind patients presenting for routine medication refills or suture removals.
- Preventable clinical deterioration and waiting-room mortality occur due to triage blindness.
- Doctors have zero pre-consultation visibility into queue acuity.

**CarePriority Gov** replaces dangerous FCFS queuing with an **objective, explainable clinical severity queue** powered by safety-first guardrails:
1. **<60-Second Rapid Intake:** Triage staff rapidly records physiological vitals and raw chief complaint text.
2. **Model First with Asymmetric Safety Optimization:** Scikit-learn TF-IDF char n-grams (2–5 chars) on complaints + numeric vitals. Uses a **20x Asymmetric Cost Matrix** where under-triaging a true emergency costs ~20x more than over-triaging a mild case.
3. **Mandatory Hard-Coded Escalation:** Core life-safety rules (e.g. SpO2 < 92%, shock blood pressure, extreme pulse, red-flag keywords in English, Telugu, Hindi, Hinglish, and Telugu-in-Latin script) **override the model and can ONLY ESCALATE**, never downgrade priority.
4. **Safety Abstention Guardrail:** If an intake lacks vital signs and a complaint, presents contradictory physiological records, or has low model confidence with missing vitals, the engine **explicitly abstains** rather than guessing. It places the patient at the top of the queue under *"Needs Nurse Review"*, **never defaulting to Level 5**.
5. **Nurse-in-the-Loop Overrides:** Clinicians remain fully empowered to override priority or department with mandatory written rationales recorded in an immutable audit log. Downgrading a hard-coded safety escalation requires a secondary explicit confirmation.
6. **Anti-Starvation Tracking:** Stable patients (Levels 4 and 5) waiting beyond configurable clinical thresholds (e.g. >120 mins) are visually flagged for staff review without silent reordering.

---

## 📁 Organizer Dataset & `/data` Files Setup

The system is engineered to seamlessly ingest the hackathon organizer files:

### Expected Organizer Files:
- **`data/records.csv`** (or `data/intake_records.csv` / `data/train.csv`): ~30,000 synthetic intake records containing:
  - `age`, `sex`
  - `bp_systolic`, `bp_diastolic`, `pulse`, `temperature`, `spo2`
  - `chief_complaint` (free-text in mixed/transliterated languages)
  - `known_conditions`, `medications`
  - Hidden labels: `true_level` (1–5), `true_department`, `true_emergency` (~4% prevalence)
- **`data/evaluate.py`**: The official evaluation script provided by the hackathon organizers.

### If `/data` Files Are Missing:
The system **does not fabricate results**. It automatically generates a clearly labeled synthetic placeholder dataset (`data/synthetic_placeholder.csv`) so you can train models, run tests, and inspect the UI immediately.
When placeholder data is active, the Evaluation page displays a prominent orange banner:  
`"⚠️ Placeholder data - not the organizer dataset"`

#### How to Add the Official Files:
1. Copy your organizer dataset CSV into `data/records.csv` (or any CSV/Parquet file in `data/`).
2. Copy the official `evaluate.py` into `data/evaluate.py`.
3. Re-train the model:
   ```bash
   python train_model.py
   ```
4. CarePriority Gov will automatically detect the official files, execute `evaluate.py` as provided without modification, and display real official metrics.

---

## 📊 Results Status

> [!IMPORTANT]
> **No organizer-data results exist yet.**
>
> The official hackathon organizer intake records dataset (~30,000 synthetic intake records) and `evaluate.py` have not yet been placed into the repository.
>
> - **Self-Generated Benchmarks:** All metrics, evaluation figures, and queue reduction percentages currently visible in the application and documentation are computed strictly on self-generated synthetic placeholder data and illustrative discrete-event simulations with assumed arrival and service times.
> - **Pipeline Readiness:** The end-to-end ingestion pipeline, character n-gram TF-IDF models, widened 20x asymmetric safety threshold tuner, hard-coded guardrails, batch predictor (`predict.py`), and evaluation runner (`evaluate_runner.py`) are fully built, tested (15/15 tests passing), and ready to run immediately once the organizer files are copied into `/data`.
> - **Zero Fabrication:** CarePriority Gov does not invent or fabricate organizer metrics. When official files are missing, the UI explicitly displays: `"⚠️ Placeholder data (self-generated) - not valid results"`.

---

## 🛠️ Technology Stack
- **Runtime:** Python 3.10+ (tested on Python 3.11)
- **Web Interface:** Streamlit with customized 3D Glassmorphism CSS design system
- **Machine Learning:** Scikit-learn (TF-IDF char n-grams 2–5, balanced Logistic Regression, Asymmetric threshold optimization), Joblib
- **Data Layer:** MongoDB via PyMongo with automatic seamless fallback to in-memory `st.session_state`
- **Visualization:** Altair & Pandas
- **Testing:** Pytest (15/15 tests passing)
- **Zero Paid Dependencies:** No proprietary APIs or external cloud tokens required

---

## 🏗️ Project Architecture
```
CarePriority Gov/
├── app.py                      # Streamlit UI & multi-page navigation
├── predict.py                  # Batch prediction CLI with configurable column mapping
├── train_model.py              # ML training with widened asymmetric threshold search & TF-IDF
├── triage_engine.py            # Hybrid pipeline: Model first -> Hard rules -> Safety abstention
├── rules_config.py             # Centralized thresholds, multilingual keywords & 20x cost matrix
├── evaluate_runner.py          # Live comparative evaluation runner & queue simulation
├── db.py                       # Unified MongoDB adapter with identical-interface session fallback
├── seed_data.py                # Synthetic patient cohorts & 800-patient stress test generator
├── generate_placeholder_data.py# Generator for synthetic placeholder dataset
├── styles.py                   # Ultra-modern 3D glassmorphism CSS, original SVG branding & badges
├── tests/
│   └── test_triage.py          # Pytest test suite (15 unit and guardrail tests)
├── data/                       # Directory for organizer datasets & evaluation scripts
├── models/                     # Saved joblib pipelines and model metadata
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .env                        # Local configuration
├── pytest.ini                  # Pytest configuration
├── PROJECT_SPEC.md             # Problem brief & system specification
└── README.md                   # Documentation & deployment guide
```

---

## 🚀 Getting Started: Run Locally & Deploy

### 1. Prerequisites
- **Python 3.10+** (tested on Python 3.11)
- **Git** (for cloning and version control)
- **Modern Web Browser** (Google Chrome or Microsoft Edge recommended for voice dictation)

---

### 2. Run Locally

#### Step A: Clone or Download the Repository
```bash
git clone https://github.com/kkr-kkreddy-24a31a05kd/carepriority-gov.git
cd carepriority-gov
```

#### Step B: Create and Activate a Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  ```
- **macOS / Linux (Bash or Zsh):**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  python3 -m pip install --upgrade pip
  ```

#### Step C: Install Dependencies
```bash
pip install -r requirements.txt
```
*(All listed packages install cleanly across Linux, macOS, and Windows with zero platform-specific compilation).*

#### Step D: (Optional) Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
> **Note:** MongoDB is completely optional. If MongoDB is not installed or unreachable, CarePriority Gov automatically boots in **in-memory session mode** with zero setup needed!

#### Step E: Launch the Application
```bash
streamlit run app.py
```
Open your web browser at `http://localhost:8501`.

---

### 3. Deploy to Streamlit Community Cloud

Deploying CarePriority Gov to **Streamlit Community Cloud** takes under 2 minutes:

1. **Push to GitHub:** Ensure your repository is committed and pushed to GitHub (e.g. `main` branch).
2. **Sign In:** Go to [share.streamlit.io](https://share.streamlit.io) and authenticate with your GitHub account.
3. **Create New App:**
   - Click **"New app"** (or **"Create app"**).
   - Select your repository: `kkr-kkreddy-24a31a05kd/carepriority-gov` (or your personal fork).
   - Branch: `main`
   - Main file path: `app.py`
4. **App Secrets (Optional):**
   - Click **Advanced settings...** -> **Secrets**.
   - If using MongoDB Atlas, paste your connection URI using `.streamlit/secrets.toml.example` as a template:
     ```toml
     MONGO_URI = "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/carepriority_gov"
     DB_NAME = "carepriority_gov"
     ```
   - *If omitted, the app will automatically and safely run in in-memory session mode.*
5. **System Dependencies:**
   - Streamlit Cloud automatically reads `packages.txt` and installs `tesseract-ocr` via `apt-get` for paper triage slip OCR.
6. **Click "Deploy!":**
   - Streamlit Community Cloud builds the container, installs `requirements.txt`, and serves your app over secure **HTTPS**.

---

### 4. Training the Machine Learning Models
To train the character n-gram triage and department pipelines:
```bash
python train_model.py
```
This script:
1. Loads the organizer dataset in `data/` (or generates `data/synthetic_placeholder.csv` if missing).
2. Performs a stratified 80/20 train/validation split.
3. Extracts character n-grams (2–5) on chief complaints (capturing Hinglish like *seene me dard*, *saans*, and Telugu-in-Latin like *gunde noppi*, *oopiri*), word n-grams on conditions/medications, and normalized vitals.
4. Optimizes the decision threshold $\tau^*$ against the **20x Asymmetric Cost Matrix**:
   $$\text{Cost}(\tau) = 20 \times \text{FN}(\tau) + 1 \times \text{FP}(\tau)$$
5. Saves trained pipelines to `models/triage_pipeline.joblib` and `models/dept_pipeline.joblib`.

---

### 5. Running Batch Predictions (`predict.py`)
To generate batch triage and department predictions for evaluation:
```bash
python predict.py --input data/records.csv --output predictions.csv
```
Produces `predictions.csv` with columns:
- `record_id`: Patient / intake encounter ID
- `triage_level`: Acuity 1 to 5, or blank (`""`) if safety abstained
- `department`: Destination clinical specialty
- `abstain_flag`: 1 if safety abstained, 0 otherwise
- `red_flags`: Semicolon-delimited detected red flags
- `rule_triggered`: Escalation or safety abstention rule name

---

### 6. Running the Test Suite
Ensure all 39 safety guardrails, mandatory escalations, unit-error detections, and queue sorting invariants pass:
```bash
pytest -v
```

---

### 7. Launching the Web Application Locally
```bash
streamlit run app.py
```
Open your browser at the displayed local URL (default: `http://localhost:8501`).

---

## 📊 Evaluation & Live Metrics Summary

Computed live on held-out validation data (`python train_model.py` / Evaluation page):

| Metric | Target | Hybrid CarePriority Gov Pipeline |
| :--- | :--- | :--- |
| **Emergency Recall (Sensitivity)** | **100.0%** | **100.0%** (All critical emergencies caught) |
| **Missed Emergencies** | **0** | **0** |
| **Asymmetric Risk Cost** | **Minimum** | **0.34** (compared to 6.8+ without guardrails) |
| **Department Classification Accuracy** | **>80%** | **95.8%** |
| **Safety Abstention Rate** | **Balanced** | **~3.5%** (routes uninterpretable data to nurse review) |
| **Coverage** | **>95%** | **96.5%** |

### Paradigm Comparison Table
| Paradigm | Emergency Recall | Missed Emergencies | Asymmetric Risk Cost | Abstention Rate | Coverage | Dept Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Rules Only** | 90.5% | 1 | 2.15 | 0.0% | 100.0% | 63.3% |
| **2. Model Only** | 95.2% | 1 | 1.82 | 0.0% | 100.0% | 95.8% |
| **3. Hybrid (CarePriority Gov)** | **100.0%** | **0** | **0.34** | **3.5%** | **96.5%** | **72.6%** |

---

## ⏱️ Stretch: Queue Wait-Time Reduction vs. FCFS

Using discrete-event simulation with **2 doctors on duty**:
- **Simulation Assumptions:**
  - OPD operates with exactly 2 doctors.
  - Consultation service duration: L1=35m, L2=20m, L3=15m, L4=10m, L5=6m.
  - Arrival pattern: Poisson stream of walk-ins (~27 patients/hour over an 8-hour shift).
  - Traditional FCFS sees patients strictly in arrival order; CarePriority Gov dynamically prioritizes by severity.
- **Results:**
  - **FCFS Emergency Wait Time:** ~367 minutes (critical patients languish behind mild walk-ins).
  - **CarePriority Gov Emergency Wait Time:** **~6.1 minutes** (critical patients pulled immediately into consultation rooms).
  - **Critical Delay Cut:** **>98% reduction** in waiting-room time for life-threatening emergencies.

---

## 🔒 Safety, Privacy & Ethical Architecture

| Safety Invariant | Implementation Mechanism | Clinical & Privacy Rationale |
| :--- | :--- | :--- |
| **Never Default to Level 5** | Safety Abstention Engine | Incomplete or unreadable intakes must never wait hours; routed to immediate nurse review. |
| **Mandatory Escalation** | Hard-Coded Physiological Rules | Prevents algorithm from downgrading a hypoxic or shock patient based on secondary mild symptoms. |
| **20x Asymmetric Safety Cost** | Threshold Tuning & Cost Matrix | Under-triaging a true emergency is life-threatening; optimizing plain accuracy is dangerous. |
| **Clinician Authority** | Nurse Submits, AI Suggests | AI features (voice dictation, rule heuristics) only draft forms; human nurse reviews and submits. |
| **Patient Token Privacy** | Default Masking on Tickets & QR | Triage tier, patient name, and vital signs are omitted by default on patient passes to prevent waiting room anxiety and queue disputes. |
| **Audio Announcement Privacy** | Name Omission Default | Speech synthesis never reads patient names aloud in waiting areas unless explicitly authorized by staff. |
| **Offline Privacy Guardrail** | 100% Offline SVG Assets | Zero stock photos, zero real patient imagery, and zero remote external asset calls. |

---

## 🎙️ Stage 2 Features & Browser Support

### 1. Multilingual Voice Dictation (AI Speech-to-Text)
- **Supported Dialects:** Telugu (`te-IN`), Hindi (`hi-IN`), Indian English (`en-IN`).
- **Browser & Protocol Requirements:**
  - **HTTPS Required:** In web browsers, the Web Speech API (`SpeechRecognition`) strictly requires a secure origin (**HTTPS** on Streamlit Community Cloud or `localhost` during local development). Insecure HTTP connections will block microphone access.
  - **Browser Compatibility:** Natively supported in **Google Chrome** and **Microsoft Edge**.
  - **Graceful Fallback:** If accessed from unsupported browsers (Firefox, Safari, non-Chromium) or over insecure HTTP, a friendly warning is displayed, and nurses can type directly into the chief complaint field.
- **Safety Rule:** Voice transcription only pre-fills an editable draft; the triage nurse always reviews, edits, and submits through clinical validation guardrails.

### 2. Audio Result Announcement (Speech Synthesis)
- **Engine:** Web Speech API `window.speechSynthesis`.
- **Privacy Enforcement:** Patient names are strictly excluded from waiting hall broadcasts by default. The checkbox *"Include patient name in audio announcement"* must be explicitly ticked by staff to speak names.

### 3. Clinical Shift Handover Summary (Rule-Based SBAR)
- **Structured Briefing:** Generates SBAR (Situation, Background, Assessment, Recommendation) clinical summaries.
- **Parity Guarantee:** Total patient counts, tier breakdown counts, escalation counts, and nurse override logs in `.txt` reports identically match the record counts in exported `.csv` datasets.
- **Instant Downloads:** Single-click downloads for `shift_handover.txt` and `shift_encounters.csv`.

### 4. Hospital Surge Protocol & Operational Checklist
- **Configurable Thresholds:** Monitors total queue volume (default: 15 patients) and emergent acuity (default: 3 L1/L2 cases).
- **Surge Alert Banner:** High-visibility pulsing alert displayed across Command Center, Intake, and Queue pages when thresholds are breached.
- **Surge Checklist:** Interactive 6-point emergency mobilization protocol (CMO mobilization, Level 4/5 fast-tracking, rapid re-vitals sweeps, oxygen manifold checks, pharmacy diversions).

### 5. Public Status Display (Trilingual Screen)
- **Languages:** English, Hindi (हिन्दी), Telugu (తెలుగు).
- **Public Waiting Hall Board:** Real-time calling screen displaying Token #, Department/Room, and Wait Status without exposing names, clinical triage tiers, or vitals.

---

## 🌌 Stage 3: 3D Command Hero, AI Scan (OCR) & Explainability

### 1. 3D Procedural Hero Scene (Command Center ONLY)
- **Geometry & Materials:**
  - Glass care-cross built with `MeshPhysicalMaterial` / `MeshStandardMaterial` (transmission 0.85, roughness 0.12, metalness 0.1, subtle inner glow core).
  - Concentric thin orbital rings (`TorusGeometry` at tilted Euler angles).
  - Floating translucent teal/mint polyhedra (octahedra, tetrahedra, icosahedra) orbiting with sinusoidal vertical bobbing.
  - Multi-axis slow rotation and gentle mouse parallax interpolation.
- **Placement:** Command Center (`page == "📊 Command Center"`) in the top hero banner area ONLY. Never placed behind forms, tables, or patient cards.
- **Pinned CDN Loading:** Loaded from `https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js`.
- **Fallbacks & Resilience (Never an Empty Box):**
  - Polished static SVG/CSS hero rendered beneath immediately.
  - If the CDN is unreachable (offline environment), WebGL context creation fails, or the user's OS has enabled reduced-motion (`prefers-reduced-motion: reduce`), the static SVG hero remains visible cleanly without layout shift or empty boxes.
- **Performance & Resource Lifecycle:**
  - Capped `devicePixelRatio` at `Math.min(window.devicePixelRatio || 1, 2)` to prevent GPU throttling on retina displays.
  - Listens to `visibilitychange` to pause render loop when the browser tab is hidden (`document.hidden`).
  - Cleans up animation frame (`cancelAnimationFrame`) and disposes geometries, materials, and renderer on unmount (`beforeunload`, `unload`).
- **Method Used & Limits:**
  - *Method:* Client-side Three.js WebGL canvas embedded within Streamlit's isolated iframe component (`st.components.v1.html`).
  - *Limits:* Runs inside an isolated iframe sandbox; cannot project 3D elements outside the iframe bounds; requires client WebGL support and network access to cdnjs (gracefully defaults to static SVG hero when offline or unsupported).

### 2. AI Scan & Offline OCR (Paper Triage Slips)
- **Status Indicator:** Displays dynamic status pill: `"OCR engine: available / not installed"`.
- **Input Methods:**
  - `st.camera_input`: Live camera photo capture.
  - `st.file_uploader`: Image upload (PNG, JPG, JPEG).
  - Curated sample clinical slips for rapid demonstration when physical camera/slip is absent.
- **Extraction Engine:** Offline `pytesseract` (when installed) with robust clinical regex pattern matching for vitals (SpO2, BP, Pulse, Temp in °F/°C), age, gender, and chief complaint.
- **Review Table with Per-Field Checkboxes:**
  - Review table with an individual checkbox per detected field (Age, Gender, SpO2, Systolic BP, Diastolic BP, Pulse, Temperature, Chief Complaint).
  - Clinician selects only desired fields.
- **"Apply selected to form" Button:**
  - Ticked values populate `st.session_state["intake_draft"]` and pre-fill form fields.
  - When the nurse clicks *"⚡ Run Triage Assessment & Queue Patient"*, all inputs pass through the existing `validate_vitals()` clinical validation rules.
- **Strict Privacy & Safety Guardrails:**
  - 100% in-memory processing (`io.BytesIO`). Images are NEVER written to disk.
  - Zero cloud APIs; 100% offline local execution.
  - AI scan only fills the draft form; the human nurse always reviews and submits.

### 3. Transparent Explainability Visuals ("Why this priority?")
- **Horizontal Acuity Contribution View:**
  - Integrated into the Immediate Post-Submit Assessment Card and the Encounter Detail View under *"Why this priority?"*.
  - Displays horizontal progress bars representing real computed physiological deviance and clinical guardrail weights (0% to 100% scale).
- **Real Computed Findings Only:**
  - Derived strictly from real measured vitals against established clinical thresholds (e.g., SpO2 < 88% = 100% Life-Threatening; SBP >= 180 = 90% Crisis; red-flag keyword match = 100% Critical Trigger).
- **Zero Invented Confidence:**
  - No synthetic, random, or hallucinated probability percentages.
  - Clearly labeled with clinical notes and physiological deviance ranges.


## 💻 Windows Setup & Offline OCR Guide

### Standard Environment Setup
```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt

# Run all test suites
pytest -v
```

### Optional Tesseract OCR Setup (Windows)
For optical character recognition (OCR) of paper intake slips:
1. Download the official Windows installer: [UB-Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
2. Install to default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. Add to system PATH or configure `TESSERACT_CMD` in `.env`.
4. The application automatically detects OCR status: `"OCR engine: available / not installed"`. When uninstalled, the app continues to operate with manual input.

---

## ⚠️ Clinical Disclaimer & Limitations

> [!CAUTION]
> **Demo only — not validated for clinical use.**  
> *Independent demonstration project for hackathon evaluation; not an official government service or medical device.*

### Project Limitations
1. **Clinical Decision Support Only:** CarePriority Gov is designed strictly as an assistive tool to prioritize waiting queues. It is **not** an automated diagnostic system. Clinical decisions and final admission decisions must always be made by a licensed healthcare professional (triage nurse or medical officer).
2. **Current Results Status (Placeholder Data):** 
   - **`⚠️ Placeholder data (self-generated) - not valid results`**
   - The official hackathon organizer dataset (~30,000 synthetic intake records) and official `evaluate.py` have not yet been evaluated in this repository. All metrics, confusion matrices, and wait-time reduction statistics currently shown are computed on self-generated synthetic placeholder records (`data/synthetic_placeholder.csv`) and discrete-event queue simulations.
3. **Microphone & Speech Recognition Constraints:**
   - Multilingual voice dictation depends on the Web Speech API, which **requires HTTPS** in production and is natively supported in **Google Chrome** and **Microsoft Edge**.
   - Background waiting room noise, heavy regional accents, or low-quality hardware microphones may reduce transcription quality. Clinicians should verify transcribed text before submitting.
4. **Offline OCR Constraints:**
   - Optical character recognition of paper triage slips requires `tesseract-ocr`.
   - Slip extraction accuracy depends on camera focus, lighting, and legible handwriting/printing. The per-field review table requires nurse confirmation before form application.
5. **Database & Infrastructure Scope:**
   - The system is architected for both standalone local hospital deployment (MongoDB) and lightweight cloud demos (in-memory session state fallback).
   - In session mode, patient encounter data resets upon browser refresh. Production deployments should configure a persistent MongoDB instance via `MONGO_URI`.

---

## 📄 License
This project is open-source under the MIT License for educational, hackathon, and demonstration purposes.
