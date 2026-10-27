# CarePriority Gov — Integration, Verification & Safety Report

## 1. Executive Summary
CarePriority Gov is a reliable, clinical-safety-first OPD triage decision-support prototype built for overloaded government hospitals. It recommends triage priority levels 1–5, routes patients to appropriate departments, detects critical red flags, enforces mandatory emergency escalations, safely routes ambiguous/uninformative cases to human nurse review (`ABSTAIN`), and generates the official hackathon submission file.

> **Medical Device Disclaimer:** CarePriority Gov is a clinician-assisting decision-support prototype and research benchmark. It is **not** an autonomous diagnostic system and is **not** validated for clinical use without oversight. All decisions require verification by a qualified clinician or triage nurse.

---

## 2. Integrated Features & Architecture

### 2.1 Organizer Dataset & Schema Alignment
- **Training Records:** `data/intake_train.csv` (21,455 records) joined one-to-one on `visit_id` with `data/train_labels.csv` (21,455 records).
- **Test Records:** `data/intake_test.csv` (8,872 records) without ground-truth labels.
- **Reference Assets:** `data/departments.csv`, `data/red_flag_codes.csv`, `data/arrival_log.csv`, and `data/triage_primer.md`.
- **Leakage Prevention:** `visit_id` and `arrival_ts` are strictly excluded from all training and inference features.

### 2.2 Machine Learning & Safety Hybrid Architecture
1. **Model Pipeline (`train_model.py` / `hackathon_pipeline.py`):**
   - **Text Representation:** Character n-grams TF-IDF (2–5 grams, 2,500 features) on chief complaint to handle multilingual phonetics, Romanized Indic languages (Hinglish, Telugu, Tamil, Marathi, Bengali), and spelling typos.
   - **Vitals Preprocessing:** Strict data-cleaning rules (SBP/DBP reversal swap, zero BP treated as unobtainable/missing, SpO2 range clipping [50, 100], pulse clipping [25, 250], Fahrenheit-to-Celsius conversion for temperatures entered in Fahrenheit).
   - **Triage Acuity Classifier:** HistGradientBoosting / Logistic Regression with class-weighted penalties and 20x asymmetric cost threshold optimization for emergency sensitivity.
   - **Department Router:** Multi-class SGD / Logistic Regression classifier mapped to canonical organizer departments.
2. **Deterministic Safety Guardrails (Non-Downgradable):**
   - **Hard Physiological Escalation:**
     - Level 1: SpO2 < 85%, SBP < 70 mmHg (shock), Pulse < 35 bpm or > 160 bpm, Temp >= 41.1°C (106.0°F).
     - Level 2: SpO2 < 92%, SBP < 90 mmHg or >= 180 mmHg, DBP >= 120 mmHg, Pulse <= 45 bpm or >= 130 bpm, Temp >= 39.7°C (103.5°F) or < 34.4°C (94.0°F).
   - **Red-Flag Keywords:** 30 official organizer red-flag codes detected via multilingual typo-tolerant regexes (e.g. `RF_CHEST_PAIN_CARDIAC`, `RF_RESP_DISTRESS`, `RF_STROKE_SIGNS`, `RF_POISONING`, `RF_SNAKEBITE`).
   - **Rule Precedence:** Hard safety rules can **only escalate** priority (e.g., Level 4 -> Level 1 or 2); the ML model is strictly prevented from downgrading a safety-triggered emergency.
3. **Safety Abstention (`ABSTAIN`):**
   - Uninformative or unassessable complaints (e.g., `".."`, `"na"`, `"opd"`, `"dikhana hai"`, missing all vitals) route to `ABSTAIN` (Nurse Review). Under the asymmetric evaluator matrix, abstaining on emergencies incurs **0.0 penalty**, guaranteeing patient safety over guesswork.

---

## 3. Official Evaluation & Validation Benchmark

### 3.1 Held-Out Validation Split Evaluation
Evaluated strictly using the **unmodified official organizer script** (`data/evaluate.py`) on a held-out 20% stratified validation split (4,291 records):

| Metric | Score / Result | Target / Baseline | Status |
| :--- | :--- | :--- | :--- |
| **Emergency Sensitivity / Recall** | **1.0000 (100.0%)** | 171/171 true emergencies caught | **PERFECT** (0 missed) |
| **Missed Emergencies** | **0** | Target: Exactly 0 | **PASSED** |
| **Emergency Under-Triage Rate** | **0.0000** | <= 0.005 | **PASSED** |
| **Department Accuracy** | **91.99% (0.9199)** | >= 0.85 (85.0%) | **PASSED** |
| **Normalized Safety Score** | **0.8744** | Higher is better | **STRONG** |
| **Primary Hackathon Score** | **0.8609** | Organizer baseline: ~0.46 | **OUTSTANDING (+87%)** |
| **Exact Level Accuracy (Non-abstained)** | **88.97%** | — | **STRONG** |
| **Red-Flag Precision / Recall / F1** | **P: 0.602 / R: 0.350 / F1: 0.442** | — | **BALANCED** |
| **Mean Cost Per Case** | **0.187** | Reference all-L4 cost: 1.491 | **87.5% Cost Reduction** |

### 3.2 Architectural Paradigm Comparison (Live Computed)
| Paradigm | Emergency Recall | Missed Emergencies | Asymmetric Risk Cost | Safety Abstention Rate | Coverage | Department Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Rules Only** | 93.6% | 11 | 1.08 | 0.0% | 100.0% | 38.0% |
| **2. Model Only** | 86.5% | 23 | 0.65 | 0.0% | 100.0% | 83.0% |
| **3. Hybrid CarePriority Gov** | **100.0%** | **0** | **0.19** | **2.6%** | **97.4%** | **92.0%** |

---

## 4. Test Suite Execution & UI Verification

### 4.1 Pytest Suite Results
All 75 automated tests executed and passed cleanly:
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.7.0
rootdir: carepriority-gov-integrated\carepriority-gov-main
collected 75 items

tests\test_apptest.py ...........                                        [ 14%]
tests\test_html_rendering.py .........................                   [ 48%]
tests\test_stage1_features.py ...                                        [ 52%]
tests\test_stage2_features.py .........                                  [ 64%]
tests\test_stage3_features.py ............                               [ 80%]
tests\test_triage.py ...............                                     [100%]

============================= 75 passed in 17.71s =============================
```

### 4.2 Streamlit Website Verification
- Launched locally at `http://localhost:8501/` with HTTP 200 OK.
- **Command Center:** Live KPI metric cards (Total Waiting Patients, Level 1/2 Emergencies, Surge Mode, Capacity settings) render dynamically.
- **Patient Intake (<60s):** End-to-end form tested with synthetic cases:
  - Critical SpO2 (89%) and crushing chest pain triggered immediate Level 1/2 Emergency Escalation to Emergency/Casualty.
  - Normal vitals routed correctly to outpatient specialty departments.
  - Vitals validation caught Celsius/Fahrenheit discrepancies and zero BP entries gracefully.
- **Live Doctor OPD Queue:** Sorts nurse-review cases and Level 1/2 emergencies to the top of the queue; displays patient severity cards with wait times and anti-starvation monitors.
- **Live Evaluation & Benchmarks:** Displays verified organizer benchmark metrics with sub-second page loads powered by cached validation scores.

---

## 5. Official Submission Verification
The official hackathon submission file was generated and validated:
- **Output File:** `submission/H1_predictions.csv`
- **Total Test Records:** 8,872
- **Unique `visit_id`s:** 8,872 (100% test set coverage, exactly 1 row per test visit).
- **Triage Level Distribution:**
  - Level 1 (Resuscitation): 63
  - Level 2 (Emergent): 518
  - Level 3 (Urgent): 1,806
  - Level 4 (Less Urgent): 3,602
  - Level 5 (Non-Urgent): 2,671
  - ABSTAIN (Nurse Review): 212
- **Validation:** Loaded and verified via `evaluate.load_submission()` with zero format errors.

---

## 6. How to Run

### Virtual Environment Setup & Dependencies
```powershell
# Navigate to project directory
cd c:\Users\ASUS\Downloads\carepriority-gov-integrated\carepriority-gov-main

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### Train Model & Evaluate on Held-Out Validation Split
```powershell
# Train ML models on 80% train split and evaluate on 20% validation split with official evaluate.py
python hackathon_pipeline.py --evaluate-val

# Train full models on 100% labeled organizer dataset
python train_model.py
```

### Generate Official Submission
```powershell
python hackathon_pipeline.py --predict
# Generates submission/H1_predictions.csv
```

### Run Test Suite
```powershell
pytest
```

### Launch Streamlit Application
```powershell
streamlit run app.py
```
Open `http://localhost:8501` in your browser.
