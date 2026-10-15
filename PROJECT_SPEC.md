# Project Specification: CarePriority Gov

## Problem Statement
**H1: OPD Triage Assistant for an Overloaded Government Hospital**

### Context
A typical government hospital Outpatient Department (OPD) handles approximately 800 walk-ins per day with only 2 doctors on duty. The traditional first-come-first-served queue is dangerous: patients with silent hypoxia, impending stroke, myocardial infarction, or severe shock wait behind mild cases, leading to preventable clinical deterioration and death in waiting rooms.

CarePriority Gov replaces the dangerous FCFS queue with a dynamic clinical severity queue:
1. A triage nurse enters vital signs and symptoms in under 60 seconds at intake.
2. The triage engine proposes a priority tier (Level 1 to Level 5) and target clinical department.
3. Hard-coded safety guardrails can only escalate (never downgrade) priority.
4. If vitals or complaint are missing or contradictory, the engine triggers a **Safety Abstention** to place the patient in a dedicated "Needs Nurse Review" category at the top of the queue.
5. The human nurse always stays in control with auditable overrides.

### Core Architectural Guardrails
- **5-Tier Scale**: L1 Resuscitation (0 min), L2 Emergent (<15 min), L3 Urgent (<45 min), L4 Less Urgent (<90 min), L5 Non-Urgent (<180 min). Labeled clearly as a demo scale inspired by 5-level triage systems — not a validated ESI implementation.
- **Mandatory Escalation**: Hard-coded physiological triggers (e.g., SpO2 < 92%, severe hypotension, pulse extremes) and red-flag symptom keywords (English, Hindi, Telugu) force L1 or L2 regardless of other scores.
- **Safety Abstention**: If data is incomplete or ambiguous, the engine abstains and routes the patient immediately to the nurse review queue. Abstention never defaults to Level 5.
- **Audit Logging**: Any nurse override of priority or department requires an explicit written rationale, capturing timestamp, actor, prior state, new state, and reason. Lowering an escalated case requires a secondary explicit confirmation.
- **Storage & Fallback**: PyMongo with auto-fallback to an identical-interface in-memory session state when MongoDB is unreachable.

### Disclaimer
*Demo only — not validated for clinical use. Independent demonstration project, not an official government service.*
