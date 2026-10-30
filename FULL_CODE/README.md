# CarePriority Gov

CarePriority Gov is a demonstration OPD intake and queueing application plus a separate adapter for the H1 synthetic triage benchmark. It is not a clinical guideline, medical device, or validated service. A qualified clinician must make all care decisions.

## What is in this project

- `app.py`, `triage_engine.py`, and `db.py`: Streamlit demonstration app, rule/model-assisted intake, nurse review and override workflow, and optional MongoDB persistence with a visible in-memory session fallback.
- `hackathon_pipeline.py`: organizer-data training, safety checks, chronological validation, and official-format prediction generation.
- `evaluate_runner.py`: organizer dataset status checks and a queue replay using the organizer's arrival and service-time records.
- `data/organizer/`: expected location for the official organizer package. The package is synthetic and contains the training labels, test intake, evaluator, baseline, primer, and arrival log.
- `submission/organizer_validation_score.json`: most recent chronological validation report produced by the organizer pipeline.

The app's demonstration models and the separate H1 organizer pipeline are distinct. Do not treat results from one as validation of the other.

## Organizer data

The organizer package provides 21,455 labeled training visits, 8,872 test visits, and an arrival log for 30,327 visits. It also supplies 13 canonical departments and 30 red-flag codes. Every record is synthetic; it does not represent a real patient.

Extract the organizer archive without changing its layout so these paths exist:

```text
data/organizer/README.md
data/organizer/evaluate.py
data/organizer/baseline/baseline.py
data/organizer/data/intake_train.csv
data/organizer/data/train_labels.csv
data/organizer/data/intake_test.csv
data/organizer/data/arrival_log.csv
data/organizer/data/triage_primer.md
```

The training schema has no patient identifier, so patient-level grouping is not possible. The organizer's labels are derived from hidden synthetic clinical state; the recorded intake can omit or misstate that state.

## Environment setup

Use Python 3.10 or later. From the project directory on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run and verify

Run the test suite:

```powershell
python -m pytest -q
```

Train/calibrate on the earlier chronological data and evaluate once on the later held-out period using the preserved official evaluator:

```powershell
python hackathon_pipeline.py --evaluate-val
```

This command trains on the earliest 70% of labeled visits, selects the emergency threshold on the next 10%, and reports the final score on the latest 20%. It writes `submission/organizer_validation_score.json`. The held-out labels are not used to select the threshold.

Generate a candidate test submission without overwriting the existing submission file:

```powershell
python hackathon_pipeline.py --predict --output submission/H1_predictions_candidate.csv
```

Prediction requires a validation report from `--evaluate-val`; it trains on all labeled training visits and uses the separately calibrated emergency threshold. The command checks test visit count, uniqueness, and original row order. The official evaluator checks format only because test labels are hidden; it cannot establish candidate accuracy.

## Hackathon archive layout

The required archive keeps this source project under `FULL_CODE/`, with `README.md` and `.gitignore` at the archive root and one prediction file at `submission/H1_predictions.csv`. After extracting the archive, enter `FULL_CODE/` before running the setup and verification commands above. The official organizer dataset is intentionally not bundled; extract its package into `FULL_CODE/data/organizer/` before running the organizer evaluation or pipeline tests.

The previously validated test prediction in this working project is `submission/H1_predictions.csv`. The command above writes a separate `submission/H1_predictions_candidate.csv` and does not overwrite that validated file. Include the verified `submission/H1_predictions.csv` at the archive path `submission/H1_predictions.csv`. The existing file passed the official format parser and exact test-row-order checks; test labels are hidden, so test-set accuracy is unknown.

Train the app's separate local models, if desired:

```powershell
python train_model.py
```

This trainer now detects the nested organizer intake/label files. It writes into `models/` and can replace the app's existing model artifacts; it is separate from the chronological official pipeline and does not produce the official submission.

Launch the Streamlit app:

```powershell
streamlit run app.py
```

The project can use session-local storage without MongoDB. Configure `MONGO_URI` and optionally `DB_NAME` through environment variables, `.env`, or Streamlit secrets to use MongoDB. Local session data is ephemeral. If a remote clear fails, the UI reports that local data was cleared but remote records may remain.

## Current organizer validation

The latest recorded chronological run used the official synthetic evaluator and produced:

| Metric | Held-out result |
|---|---:|
| Primary score | 0.846985 |
| Emergency recall | 161 / 164 (98.17%) |
| Missed emergencies | 3 |
| Non-emergency visits escalated to Level 1 or 2 | 214 |
| Department accuracy | 90.09% |
| Red-flag micro F1 | 0.5495 |
| Exact tier accuracy among non-abstained cases | 87.41% |
| Organizer baseline primary score | 0.4503 |
| Organizer baseline emergency recall | 53.05% |

There are 4,291 chronologically later held-out visits. The emergency threshold was 0.332362, selected on 2,146 separate calibration visits, where the primary score was 0.834789 and emergency recall was 95.45%. The current held-out report records three missed emergencies. This is a benchmark result, not evidence of clinical safety; the synthetic prototype must not be used to triage real patients.

Reproduce the report with `python hackathon_pipeline.py --evaluate-val`. Older score files are not current results and are not used as substitutes for this report.

## Queue replay

The benchmark page replays 21,455 labeled organizer training visits using their recorded arrival times and consultation durations. In the current two-doctor replay, the official FCFS schedule is reproduced with a mean absolute start-time error of 0 seconds. Mean emergency wait is 34.0 minutes under FCFS and 0.3 minutes in a non-preemptive priority replay; overall mean wait is 34.7 and 34.9 minutes, respectively.

This is an **oracle counterfactual**, not an estimate of the app's or model's impact: priority ordering uses organizer ground-truth triage labels, not model predictions. Its results describe synthetic benchmark records only. Current patient queue entries show elapsed wait; no patient-specific estimated service time is inferred.

## Safety and limitations

- Rule-based escalations and abstention provide decision support only; they can be incomplete, over-trigger, or miss emergencies. The current measured holdout missed 3 of 164 emergencies and over-escalated 214 non-emergency visits to Level 1 or 2.
- The model is trained and scored on synthetic organizer data. Dataset labels use hidden synthetic state, while real intake can be incomplete, inaccurate, multilingual, or out of distribution.
- The official evaluator is a benchmark, not a clinical validation protocol.
- Queue replay uses ground-truth labels and must not be represented as model performance.
- Nurse review and written override rationale do not make automated recommendations clinically validated.
- MongoDB availability is optional; session fallback is temporary and is not durable storage.
- Speech recognition depends on browser support and permissions. OCR depends on local Tesseract and requires staff verification.
- OCR needs the Tesseract executable installed separately; `packages.txt` declares `tesseract-ocr` for Linux-based Streamlit deployments.

## AI and third-party code disclosure

The organizer pipeline uses scikit-learn classifiers and deterministic rules. The Streamlit app includes local speech/OCR and optional database integrations; it is not configured to call a remote generative-AI service. Code changes made during this audit were assisted by GitHub Copilot in VS Code.

The original authorship and creation dates of the pre-existing project files cannot be verified from this checkout because it has no Git metadata or provenance record. Before submitting, the team must complete the organizer's AI/pre-built-code disclosure with the actual team history. The organizer-provided baseline and evaluator are retained under `data/organizer/`; Python dependencies are listed in `requirements.txt`.
