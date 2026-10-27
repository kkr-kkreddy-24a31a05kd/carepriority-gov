# 🏥 CarePriority Gov
### Dynamic Clinical Severity OPD Triage Assistant for Overloaded Government Hospitals

Live demo: https://carepriority-gov.streamlit.app/

> **Hackathon Problem H1:** Outpatient Department (OPD) Triage Assistant for an Overloaded Government Hospital  
> **Clinical Disclaimer:** *Demo only — not validated for clinical use. Independent demonstration project for hackathon evaluation; not an official government service or medical device.*

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
4. **Safety Abstention Guardrail (`ABSTAIN`):** If an intake lacks vital signs and a complaint, presents contradictory physiological records, or has low model confidence with missing vitals, the engine **explicitly abstains** rather than guessing. It places the patient at the top of the queue under *"Needs Nurse Review"*, **never defaulting to Level 5**. Under the official organizer cost matrix, abstaining on emergencies incurs **0.0 penalty**, prioritizing patient safety over guesswork.
5. **Nurse-in-the-Loop Overrides:** Clinicians remain fully empowered to override priority or department with mandatory written rationales recorded in an immutable audit log. Downgrading a hard-coded safety escalation requires a secondary explicit confirmation.
6. **Anti-Starvation Tracking:** Stable patients (Levels 4 and 5) waiting beyond configurable clinical thresholds (e.g. >120 mins) are visually flagged for staff review without silent reordering.

---

## 📊 Official Organizer Evaluation & Validation Benchmark

Evaluated directly using the official, unmodified organizer evaluation script (`data/evaluate.py`) on a 20% held-out stratified validation split (**4,291 records**):

```
================================================================
  H1 OPD TRIAGE - EVALUATION (Official Organizer Benchmark)
================================================================
  EMERGENCY RECALL (sensitivity): 1.0000   (171/171 true emergencies caught)
================================================================
  PRIMARY SCORE: 0.8609   (range [-0.6, 1.0], higher is better)
    = 0.6 x safety 0.8744 + 0.2 x emergency recall 1.0000 + 0.1 x department accuracy 0.9199 + 0.1 x red-flag F1 0.4424
  Safety cost: team 803.5 vs reference (all level 4) 6397.0
    missed emergencies 0.0 | under-triage 164.0 | over-triage 572.0 | abstain 67.5
  Abstained: 111 (of which 62 were truly unassessable; 86 unassessable cases exist)
  Exact level accuracy (non-abstained): 0.8897   red-flag P/R: 0.602/0.350
================================================================
```

| Metric | Measured Score | Benchmark / Target | Clinical Significance |
| :--- | :--- | :--- | :--- |
| **Emergency Sensitivity (Recall)** | **1.0000 (100.0%)** | 171/171 emergencies caught | **Zero missed emergencies** in validation cohort |
| **Emergency Under-triage Rate** | **0.0000** | $\le 0.005$ | Eliminates dangerous waiting-room discharge risk |
| **Department Classification Accuracy** | **91.99%** | $\ge 85.0\%$ | Accurate routing across 13 canonical specialties |
| **Normalized Safety Score** | **0.8744** | Higher is better | Severe reduction in asymmetric clinical risk |
| **Primary Hackathon Score** | **0.8609** | Baseline: ~0.46 | **+87% improvement** over competition starter baseline |
| **Exact Level Accuracy (Non-abstained)** | **88.97%** | — | High tier fidelity across Levels 1 to 5 |
| **Red-Flag Precision / Recall / F1** | **P: 0.602 / R: 0.350 / F1: 0.442** | — | Balanced detection across 30 red-flag conditions |
| **Mean Risk Cost Per Patient** | **0.187** | Reference all-L4: 1.491 | **87.5% reduction in patient safety risk cost** |

### Paradigm Comparison Table (Live Computed)
| Paradigm | Emergency Recall | Missed Emergencies | Asymmetric Risk Cost | Safety Abstention Rate | Coverage | Dept Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Rules Only** | 93.6% | 11 | 1.08 | 0.0% | 100.0% | 38.0% |
| **2. Model Only** | 86.5% | 23 | 0.65 | 0.0% | 100.0% | 83.0% |
| **3. Hybrid (CarePriority Gov)** | **100.0%** | **0** | **0.19** | **2.6%** | **97.4%** | **92.0%** |

---

## 📁 Organizer Dataset & Verification

The system is trained and evaluated on the official hackathon organizer competition files:

### Organizer Dataset Assets:
- **`data/intake_train.csv` & `data/train_labels.csv`:** 21,455 labeled training records joined one-to-one on `visit_id`.
- **`data/intake_test.csv`:** 8,872 unlabeled test records.
- **`data/evaluate.py`:** Unmodified official scoring script.
- **`data/departments.csv`:** 13 canonical hospital departments.
- **`data/red_flag_codes.csv`:** 30 official emergency red-flag diagnostic codes.
- **`data/arrival_log.csv` & `data/triage_primer.md`:** Arrival timestamps and clinical escalation definitions.

### Data Leakage Controls:
- `visit_id` and `arrival_ts` are strictly excluded from all training and inference features.
- Testing data (`intake_test.csv`) has no ground-truth labels and is used strictly for prediction.
- No test accuracy or hidden-test scores are fabricated; all reported metrics derive strictly from the held-out 20% validation split evaluated by `data/evaluate.py`.

---

## 📄 Official Submission (`submission/H1_predictions.csv`)

The official prediction file has been generated and validated with `data/evaluate.py`'s `load_submission()` parser:
- **Location:** `submission/H1_predictions.csv`
- **Total Records:** 8,872 (100% coverage of all test visits in exact order).
- **Columns:** `visit_id`, `triage_level`, `department`, `red_flags`, `confidence`.
- **Distribution:** Level 1: 63, Level 2: 518, Level 3: 1,806, Level 4: 3,602, Level 5: 2,671, ABSTAIN: 212.
- **Department Mapping:** All 8,872 rows adhere strictly to the 13 canonical department names.

---

## 🛠️ Technology Stack
- **Runtime:** Python 3.10+ (tested on Python 3.11 and 3.12)
- **Web Interface:** Streamlit with customized 3D Glassmorphism CSS design system
- **Machine Learning:** Scikit-learn (TF-IDF char n-grams 2–5, class-weighted HistGradientBoosting / Logistic Regression, Asymmetric threshold optimization), Joblib
- **Data Layer:** MongoDB via PyMongo with automatic seamless fallback to in-memory `st.session_state`
- **Visualization:** Altair & Pandas
- **Testing:** Pytest (**75/75 tests passing**)
- **Zero Paid Dependencies:** No proprietary APIs or external cloud tokens required

---

## 🏗️ Project Architecture
```
CarePriority Gov/
├── app.py                      # Streamlit UI & multi-page navigation (7 pages)
├── hackathon_pipeline.py       # Official competition training, evaluation, & submission generator
├── train_model.py              # ML training with asymmetric threshold search & TF-IDF
├── triage_engine.py            # Hybrid pipeline: Model first -> Hard rules -> Safety abstention
├── rules_config.py             # Centralized thresholds, multilingual keywords & 20x cost matrix
├── evaluate_runner.py          # Live comparative evaluation runner with precomputed cache
├── explainability.py           # Acuity factor cards and transparent decision tree visualization
├── db.py                       # Unified MongoDB adapter with identical-interface session fallback
├── seed_data.py                # Synthetic patient cohorts & 800-patient stress test generator
├── smart_ticket.py             # Digital QR token generation and PDF/print view
├── ai_scan.py                  # Offline OCR paper triage slip scanner
├── audio_speech.py             # Multilingual voice dictation & speech synthesis
├── hero_3d.py                  # Procedural Three.js 3D command scene
├── surge_mode.py               # Mass-casualty & hospital surge protocols
├── shift_handover.py           # Clinician SBAR shift briefing generator
├── styles.py                   # Ultra-modern 3D glassmorphism CSS, original SVG branding & badges
├── tests/                      # Automated test suite (6 test files, 75 tests)
├── data/                       # Directory for organizer datasets, evaluator, and cached scores
├── models/                     # Saved production joblib pipelines and model metadata
└── submission/                 # Official H1_predictions.csv and validation score reports
```

---

## 🚀 Quickstart & Setup Guide

### 1. Local Environment Setup (Windows PowerShell)
```powershell
# Clone repository
git clone https://github.com/kkr-kkreddy-24a31a05kd/carepriority-gov.git
cd carepriority-gov

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Local Setup (Linux / macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Automated Test Suite
```bash
pytest
```
*Expected: 75 passed.*

### 4. Train the ML Models & Evaluate
```bash
# Train models on 80% split and score 20% validation with official evaluate.py:
python hackathon_pipeline.py --evaluate-val

# Train full models on 100% labeled organizer dataset:
python train_model.py
```

### 5. Generate Official Hackathon Submission
```bash
python hackathon_pipeline.py --predict
# Generates submission/H1_predictions.csv
```

### 6. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deploy to Streamlit Community Cloud

Deploying CarePriority Gov to **Streamlit Community Cloud** takes under 2 minutes:

1. **Fork or Push to GitHub:** Ensure your repository is on the `main` branch.
2. **Sign In:** Go to [share.streamlit.io](https://share.streamlit.io) and authenticate with your GitHub account.
3. **Create New App:**
   - Select your repository: `kkr-kkreddy-24a31a05kd/carepriority-gov`
   - Branch: `main`
   - Main file path: `app.py`
4. **App Secrets (Optional):**
   - Click **Advanced settings...** -> **Secrets**.
   - If using MongoDB Atlas, configure `.streamlit/secrets.toml`:
     ```toml
     MONGO_URI = "mongodb+srv://<username>:<password>@<cluster>.mongodb.net/carepriority_gov"
     DB_NAME = "carepriority_gov"
     ```
   - *If omitted, the application runs in zero-configuration in-memory session mode.*
5. **System Dependencies:**
   - Streamlit Cloud automatically reads `packages.txt` and installs `tesseract-ocr` via `apt-get` for paper triage slip OCR.
6. **Click "Deploy!":**
   - Streamlit Community Cloud builds the container, installs `requirements.txt`, and serves your app over secure **HTTPS**.

---

## ⏱️ Queue Wait-Time Reduction vs. FCFS (Discrete-Event Simulation)

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

## 🎙️ Advanced Demonstration Features

1. **Multilingual Voice Dictation:** Supports Telugu (`te-IN`), Hindi (`hi-IN`), Indian English (`en-IN`) via the Web Speech API on secure origins (HTTPS or localhost).
2. **Audio Result Announcement:** Speech synthesis announcing token callouts without exposing patient names or triage tiers.
3. **Clinical Shift Handover (SBAR):** Generates structured SBAR handover summaries with verified numerical parity across text and CSV exports.
4. **Hospital Surge Protocol Checklist:** Monitors queue volume and emergent acuity; triggers surge alert banners and interactive 6-point mobilization protocols.
5. **Trilingual Public Display:** English, Hindi (हिन्दी), and Telugu (తెలుగు) calling screen for patient waiting halls.
6. **3D Procedural Hero Scene:** Lightweight Three.js procedural glass care-cross rendered strictly in the Command Center with clean static SVG fallback when offline.
7. **Offline OCR Slip Scanner:** Paper slip image extraction via `pytesseract` with individual field confirmation checkboxes.
8. **Explainability Factor Cards:** Real computed physiological deviance progress bars explaining priority drivers without fabricated confidence scores.

---

## ⚠️ Clinical Disclaimer & Limitations

> [!CAUTION]
> **Demo only — not validated for clinical use.**  
> *Independent demonstration project for hackathon evaluation; not an official government service or medical device.*

### Project Limitations:
1. **Decision Support Only:** CarePriority Gov is designed strictly as an assistive tool to prioritize waiting queues. It is **not** an automated diagnostic system. Clinical decisions and final admission decisions must always be made by a licensed healthcare professional.
2. **Synthetic Organizer Data:** All training and test records are synthetic datasets provided for the hackathon competition.
3. **Web Speech API Constraints:** Voice dictation requires a secure origin (HTTPS in production, localhost in development) and Chromium-based browsers (Chrome, Edge).
4. **OCR Accuracy:** Optical slip recognition depends on camera lighting and handwriting legibility; requires nurse review before applying.

---

## 📄 License
This project is open-source under the MIT License for educational, hackathon, and demonstration purposes.
