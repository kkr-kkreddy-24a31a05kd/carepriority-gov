# Integration and Verification Report

## Scope

The official synthetic H1 organizer package is preserved under `data/organizer/`. The project has a separate Streamlit demonstration app and organizer benchmark pipeline; their models and reported results are not interchangeable. This report describes local verification only and does not establish clinical safety.

## Organizer data and pipeline

- 21,455 intake records join one-to-one with 21,455 labels; the official test intake has 8,872 visits.
- The arrival log has 30,327 visits and recorded consultation minutes and FCFS consultation starts.
- The organizer evaluator and baseline are used from the preserved package; organizer inputs, evaluator, and baseline are not modified by the adapter.
- Organizer training excludes `visit_id` and `arrival_ts` as model features and uses the organizer baseline's vital cleaning.
- Chronological evaluation uses the earliest 70% for training, the next 10% for threshold calibration, and the latest 20% for held-out scoring. The dataset schema has no patient identifier for patient-level grouping.

## Latest official validation

The latest run used 15,018 training visits, 2,146 calibration visits, and 4,291 chronologically later held-out visits. The emergency threshold was selected on calibration data and evaluated with the unmodified organizer evaluator.

| Metric | Result |
|---|---:|
| Primary score | 0.846985 |
| Emergency recall | 161 / 164 (0.981707) |
| Missed emergencies | 3 |
| Non-emergency visits escalated to Level 1 or 2 | 214 |
| Department accuracy | 0.900927 |
| Red-flag micro F1 | 0.549521 |
| Red-flag precision / recall | 0.754386 / 0.432161 |
| Exact tier accuracy among non-abstained visits | 0.874142 |
| Organizer baseline primary score | 0.450313 |
| Organizer baseline emergency recall | 0.530488 |

The calibrated emergency threshold was 0.332362. On the separate calibration split, primary score was 0.834789 and emergency recall was 0.954545. The held-out score uses the unmodified organizer evaluator and must not be interpreted as clinical validation. `submission/organizer_validation_score.json` records the full report.

### Review of the earlier 13 missed emergencies

The previous model/rule set under-triaged the following holdout cases. Rule gaps were corrected where supported by the organizer primer and checked against the separate calibration split. These identifiers are listed for auditability; holdout labels were not used to select the probability threshold.

| Visit ID | Observed gap in the earlier prediction | Current rule coverage |
|---|---|---|
| OPD-OTGDQS7V | Romanized Hindi fever with repeated vomiting and dizziness; dengue-warning wording was not recognized. | Added multilingual fever and warning-sign phrases. |
| OPD-LGJY5IJI | Tamil fever, abdominal pain, fatigue, and dizziness; dengue-warning wording was not recognized. | Added Tamil fever and abdominal-pain phrases. |
| OPD-3M62JEA7 | Generic febrile cough; recorded pulse was amber and SpO2 was missing, with no labeled red flag in the intake. | Still missed. The hidden emergency cause is not observable in the recorded fields. |
| OPD-PTCF7Z0B | Sudden loss of vision in one eye described in Romanized Hindi. | Added sudden monocular vision-loss detection. |
| OPD-JVZ4WMLZ | Sudden loss of hand sensation described in Hindi. | Added Hindi stroke-symptom detection. |
| OPD-WCLN97ML | Large-volume blood with cough (“katori bhar”). | Added hemoptysis and large-volume bleeding detection. |
| OPD-0E5XE4TZ | Heavy feeling in the center of the chest with profuse sweating. | Added cardiac chest-heaviness and sweating detection. |
| OPD-ALKKJBMK | Older adult with diabetes, uneasiness, and stomach burning, a cardiac-equivalent presentation. | Added context-dependent cardiac-equivalent detection. |
| OPD-TIQZUKP4 | Diabetes with weakness, sweating, breathlessness, and dizziness in Telugu transliteration. | Added multilingual cardiac-equivalent markers. |
| OPD-I27AST6Y | Diabetes with weakness, cold sweat, and dizziness in Hindi; the nukta spelling in “कमज़ोरी” is not currently matched. | Still missed. No additional spelling rule was added without independent calibration evidence. |
| OPD-YK34X4FP | Symptoms during pesticide spraying; transliterated pesticide term was not recognized. | Added transliteration coverage for pesticide exposure. |
| OPD-7TQHEMC6 | Major burn described in Bengali transliteration. | Added the observed transliterated burn phrase. |
| OPD-DRWUF9BB | Reported very high blood pressure and headache, but the systolic and diastolic measurements were missing. | Still missed. The input lacks the measurement needed to apply the primer’s vital threshold. |

The current three missed cases are OPD-3M62JEA7, OPD-I27AST6Y, and OPD-DRWUF9BB. The remaining errors demonstrate limitations from unobserved synthetic state, multilingual matching, and missing measurements; emergency recall is not 100%.

## Submission candidate

`submission/H1_predictions_candidate.csv` was generated using the calibrated organizer pipeline and accepted by the official submission parser:

- 8,872 rows, all with unique visit IDs matching the official test order.
- Canonical submission columns and departments were accepted by `evaluate.load_submission()`.
- The existing `submission/H1_predictions.csv` was not overwritten. Test labels are unavailable, so candidate test performance cannot be measured locally.

## Arrival-log queue replay

The benchmark page replays 21,455 labeled training visits using recorded arrivals and consultation durations. With two non-preemptive doctors, the FCFS replay matches the organizer FCFS start log with 0.00 seconds mean absolute error. Mean emergency wait is 34.0 minutes for FCFS and 0.3 minutes for a tier-priority replay; overall mean wait is 34.7 and 34.9 minutes, respectively.

This is an oracle counterfactual using ground-truth labels, not model predictions or an estimate of actual CarePriority performance. It describes synthetic records only. Current app queue entries display elapsed wait rather than patient-specific estimated service time.

## Persistence and application checks

- MongoDB write, read, update, and clear failures switch to session storage and set an explicit local-mode status.
- A failed remote clear returns failure; the app aborts demo/stress-test replacement and tells staff remote records may remain.
- Full automated suite: **125 passed** (`python -m pytest -q`).
- Python compilation passed for the changed application, database, pipeline, evaluation, and regression-test modules.
- Streamlit `AppTest` page coverage is part of the full suite. No independent clinical or production deployment verification was performed.

## Safety and disclosure limits

All organizer data is synthetic, and the primer is explicitly not for clinical use. The current holdout missed 3 of 164 emergencies and incorrectly escalated 214 non-emergency visits to Level 1 or 2. Red-flag recall and department/tier routing also have measurable error. Rule thresholds, text matching, incomplete vitals, model calibration, and distribution shift all remain risks.

The organizer baseline and evaluator are organizer-provided assets. This audit's code changes were assisted by GitHub Copilot in VS Code. This checkout has no Git metadata or provenance record, so the authorship/date of pre-existing project code cannot be certified. The team must provide the organizer-required pre-built-code and AI-assistance disclosure based on its actual history before submission.
