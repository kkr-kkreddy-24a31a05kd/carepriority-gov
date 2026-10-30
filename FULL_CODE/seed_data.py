"""CarePriority Gov - Synthetic Patient Data & Stress Test Generator
==================================================================
ALL DATA IN THIS MODULE IS STRICTLY SYNTHETIC AND ARTIFICIALLY GENERATED.
NO REAL PATIENT DATA, HEALTH IDENTIFIERS, OR CLINICAL RECORDS ARE USED.

Features:
1. Curated realistic synthetic walk-in cohort (English, Hindi, Telugu)
2. 800-patient high-volume stress test generator (simulating a full 800 walk-ins/day OPD)
3. Standard labeled evaluation benchmark set for the Evaluation Page.
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List, Any
from triage_engine import run_triage

# Curated Synthetic Patient Cohort
SYNTHETIC_COHORT_SEED: List[Dict[str, Any]] = [
    {
        "encounter_id": "SYN-20261014-001",
        "patient_name": "[SYNTHETIC] Ramesh Kumar",
        "age": 58,
        "gender": "Male",
        "spo2": 82.0,
        "bp_systolic": 85,
        "bp_diastolic": 55,
        "pulse": 142,
        "temperature": 99.1,
        "complaint_original": "Severe gasping, collapsed at reception, turning blue",
        "known_conditions": "COPD, Smoker",
        "language_hint": "en"
    },
    {
        "encounter_id": "SYN-20261014-002",
        "patient_name": "[SYNTHETIC] Lakshmi Devi",
        "age": 62,
        "gender": "Female",
        "spo2": 95.0,
        "bp_systolic": 195,
        "bp_diastolic": 115,
        "pulse": 104,
        "temperature": 98.4,
        "complaint_original": "ఛాతీ నొప్పి, ఎడమ చెయ్యి లాగుతోంది, చెమటలు (Chest pain radiating to left arm, sweating)",
        "known_conditions": "Hypertension, Type 2 Diabetes",
        "language_hint": "te"
    },
    {
        "encounter_id": "SYN-20261014-003",
        "patient_name": "[SYNTHETIC] Sunita Verma",
        "age": 45,
        "gender": "Female",
        "spo2": 89.0,
        "bp_systolic": 110,
        "bp_diastolic": 70,
        "pulse": 118,
        "temperature": 102.8,
        "complaint_original": "सांस लेने में बहुत तकलीफ हो रही है और तेज़ बुखार है (Severe breathing difficulty and high fever)",
        "known_conditions": "Asthma",
        "language_hint": "hi"
    },
    {
        "encounter_id": "SYN-20261014-004",
        "patient_name": "[SYNTHETIC] Raju Naik",
        "age": 34,
        "gender": "Male",
        "spo2": 97.0,
        "bp_systolic": 110,
        "bp_diastolic": 75,
        "pulse": 125,
        "temperature": 98.6,
        "complaint_original": "పాము కాటు కాలినొప్పి, వికారం (Snake bite on lower leg, nausea)",
        "known_conditions": "None",
        "language_hint": "te"
    },
    {
        "encounter_id": "SYN-20261014-005",
        "patient_name": "[SYNTHETIC] Mohammad Arif",
        "age": 70,
        "gender": "Male",
        "spo2": 96.0,
        "bp_systolic": 165,
        "bp_diastolic": 95,
        "pulse": 88,
        "temperature": 98.6,
        "complaint_original": "मुंह टेढ़ा हो गया है और दाहिना हाथ काम नहीं कर रहा (Sudden facial droop and right arm paralysis)",
        "known_conditions": "Previous TIA",
        "language_hint": "hi"
    },
    {
        "encounter_id": "SYN-20261014-006",
        "patient_name": "[SYNTHETIC] Master Aarav (Baby)",
        "age": 4,
        "gender": "Male",
        "spo2": 96.0,
        "bp_systolic": None,
        "bp_diastolic": None,
        "pulse": 132,
        "temperature": 103.2,
        "complaint_original": "High fever, crying continuously, vomiting after eating",
        "known_conditions": "None",
        "language_hint": "en"
    },
    {
        "encounter_id": "SYN-20261014-007",
        "patient_name": "[SYNTHETIC] Venkat Rao",
        "age": 42,
        "gender": "Male",
        "spo2": 98.0,
        "bp_systolic": 130,
        "bp_diastolic": 85,
        "pulse": 84,
        "temperature": 98.4,
        "complaint_original": "Fall from two-wheeler, severe swelling and deformity in right wrist",
        "known_conditions": "None",
        "language_hint": "en"
    },
    {
        "encounter_id": "SYN-20261014-008",
        "patient_name": "[SYNTHETIC] Geeta Bai",
        "age": 28,
        "gender": "Female",
        "spo2": 99.0,
        "bp_systolic": 115,
        "bp_diastolic": 70,
        "pulse": 86,
        "temperature": 98.6,
        "complaint_original": "గర్భం 7వ నెల, పొత్తికడుపులో నొప్పి (Pregnancy 7th month, lower abdominal cramps)",
        "known_conditions": "Primigravida",
        "language_hint": "te"
    },
    {
        "encounter_id": "SYN-20261014-009",
        "patient_name": "[SYNTHETIC] Manoj Tiwari",
        "age": 31,
        "gender": "Male",
        "spo2": 98.0,
        "bp_systolic": 120,
        "bp_diastolic": 80,
        "pulse": 76,
        "temperature": 99.2,
        "complaint_original": "गले में खराश, सूखी खांसी दो दिन से (Sore throat and mild cough for 2 days)",
        "known_conditions": "None",
        "language_hint": "hi"
    },
    {
        "encounter_id": "SYN-20261014-010",
        "patient_name": "[SYNTHETIC] Ananya Sen",
        "age": 22,
        "gender": "Female",
        "spo2": 99.0,
        "bp_systolic": 110,
        "bp_diastolic": 70,
        "pulse": 72,
        "temperature": 98.2,
        "complaint_original": "Routine blood pressure checkup and monthly iron folic acid refill",
        "known_conditions": "Mild anemia",
        "language_hint": "en"
    },
    {
        "encounter_id": "SYN-20261014-011",
        "patient_name": "[SYNTHETIC] Patient Unknown (Incomplete)",
        "age": None,
        "gender": "Other",
        "spo2": None,
        "bp_systolic": None,
        "bp_diastolic": None,
        "pulse": None,
        "temperature": None,
        "complaint_original": "",
        "known_conditions": None,
        "language_hint": "en"
    },
    {
        "encounter_id": "SYN-20261014-012",
        "patient_name": "[SYNTHETIC] Suresh Patel",
        "age": 50,
        "gender": "Male",
        "spo2": 98.0,
        "bp_systolic": 124,
        "bp_diastolic": 82,
        "pulse": 78,
        "temperature": 98.4,
        "complaint_original": "Post-operative surgical suture removal after appendectomy",
        "known_conditions": "Post-op Day 10",
        "language_hint": "en"
    }
]


def create_encounter_record(
    raw_patient: Dict[str, Any],
    minutes_ago: int = 10,
    status: str = "Waiting"
) -> Dict[str, Any]:
    """
    Runs the pure triage engine on a synthetic patient intake
    and builds an auditable, MongoDB-ready encounter document.
    """
    triage_result = run_triage(
        patient_id=raw_patient["encounter_id"],
        patient_name=raw_patient.get("patient_name"),
        age=raw_patient.get("age"),
        gender=raw_patient.get("gender"),
        spo2=raw_patient.get("spo2"),
        bp_systolic=raw_patient.get("bp_systolic"),
        bp_diastolic=raw_patient.get("bp_diastolic"),
        pulse=raw_patient.get("pulse"),
        temperature=raw_patient.get("temperature"),
        complaint=raw_patient.get("complaint_original"),
        known_conditions=raw_patient.get("known_conditions")
    )

    created_time = datetime.now() - timedelta(minutes=minutes_ago)
    
    # Set status: if abstained, force 'Awaiting Nurse Review'
    final_status = "Awaiting Nurse Review" if triage_result["abstained"] else status

    return {
        "encounter_id": raw_patient["encounter_id"],
        "created_at": created_time.isoformat(),
        "patient_name": raw_patient.get("patient_name") or f"[SYNTHETIC] Walk-in {raw_patient['encounter_id']}",
        "age": raw_patient.get("age"),
        "gender": raw_patient.get("gender", "Unknown"),
        "vitals": {
            "spo2": raw_patient.get("spo2"),
            "bp_systolic": raw_patient.get("bp_systolic"),
            "bp_diastolic": raw_patient.get("bp_diastolic"),
            "pulse": raw_patient.get("pulse"),
            "temperature": raw_patient.get("temperature")
        },
        "complaint_original": raw_patient.get("complaint_original", ""),
        "known_conditions": raw_patient.get("known_conditions"),
        "language_hint": raw_patient.get("language_hint", "en"),
        "tier": triage_result["tier"],
        "tier_name": triage_result["tier_name"],
        "department": triage_result["department"],
        "abstained": triage_result["abstained"],
        "abstention_reason": triage_result["abstention_reason"],
        "red_flags": triage_result["red_flags"],
        "reasons": triage_result["reasons"],
        "warnings": triage_result.get("warnings", []),
        "is_escalated": triage_result["is_escalated"],
        "rule_triggered": triage_result["escalation_rule"],
        "status": final_status,
        "is_synthetic": True,
        "audit": [
            {
                "timestamp": created_time.isoformat(),
                "actor": "CarePriority Triage Engine",
                "action": "Initial Intake Assessment",
                "tier": triage_result["tier"],
                "department": triage_result["department"],
                "reason": "Automated clinical rules evaluation"
            }
        ]
    }


def generate_curated_demo_records() -> List[Dict[str, Any]]:
    """Generates the clean initial synthetic cohort of ~12 representative patients."""
    records = []
    # Stagger arrival times from 5 mins ago to 120 mins ago
    for idx, raw in enumerate(SYNTHETIC_COHORT_SEED):
        mins_ago = 5 + (idx * 8)
        status = "Waiting"
        if idx == 0:
            status = "Waiting"  # L1 waiting
        elif idx == 9:
            status = "Completed"
        elif idx == 6:
            status = "In Consultation"
            
        record = create_encounter_record(raw, minutes_ago=mins_ago, status=status)
        records.append(record)
    return records


def generate_800_patient_stress_test() -> List[Dict[str, Any]]:
    """
    Generates a full 800-patient synthetic OPD cohort simulating a heavy day:
    ~5% Level 1 Resuscitation (severe shock, cardiac, stridor, arrest)
    ~15% Level 2 Emergent (hypoxia <92%, chest pain, stroke, poisoning, severe fever)
    ~35% Level 3 Urgent (fractures, pyrexia, multi-system borderline)
    ~30% Level 4 Less Urgent (sprains, URI, rash, back pain)
    ~15% Level 5 Non-Urgent (dressings, refills, checkups)
    Includes ~3% Safety Abstentions (missing vitals, ambiguous entries)
    All strictly labeled [SYNTHETIC].
    """
    random.seed(42)  # Deterministic generation
    records: List[Dict[str, Any]] = []

    names_male = ["Rajesh", "Amit", "Kiran", "Suresh", "Manoj", "Pradeep", "Vikram", "Anil", "Deepak", "Ganesh", "Mahesh", "Praveen", "Sai Krishna", "Venkat", "Harish"]
    names_female = ["Lakshmi", "Sunita", "Pooja", "Meena", "Shanti", "Kavita", "Ananya", "Swapna", "Bhavani", "Geeta", "Renuka", "Padma", "Radha", "Divya", "Suma"]
    surnames = ["Sharma", "Verma", "Rao", "Patel", "Reddy", "Gupta", "Singh", "Kumar", "Naik", "Prasad", "Yadav", "Joshi", "Mishra", "Choudhury", "Babu"]

    # Complaints pool
    l1_complaints = [
        ("Severe gasping, collapsed at reception, turning blue", "en"),
        ("Active tonic-clonic seizure for >10 mins, unresponsive", "en"),
        ("सांस नहीं आ रही, दिल रुक गया लगता है", "hi"),
        ("ఊపిరాడటం లేదు, స్పృహ కోల్పోయాడు", "te"),
        ("Massive projectile hematemesis, pulseless extremity", "en")
    ]

    l2_complaints = [
        ("Crushing retrosternal chest pain radiating to jaw", "en"),
        ("छाती में तेज़ दर्द और पसीना आ रहा है", "hi"),
        ("గుండె నొప్పి, ఎడమ చెయ్యి లాగుతోంది", "te"),
        ("Sudden facial droop and right sided paralysis", "en"),
        ("పాము కాటు, కాలు వాపు, వికారం", "te"),
        ("Accidental pesticide poisoning ingestion", "en"),
        ("Severe head trauma with altered sensorium", "en"),
        ("Profuse arterial bleeding from laceration", "en")
    ]

    l3_complaints = [
        ("High fever 102F for 4 days with chills and severe body ache", "en"),
        ("Persistent vomiting and inability to keep fluids down", "en"),
        ("तेज़ बुखार और पेट में तेज़ दर्द", "hi"),
        ("తీవ్ర జ్వరం, వాంతులు, కడుపు నొప్పి", "te"),
        ("Suspected closed ankle fracture after slip on stairs", "en"),
        ("Asthma attack partially relieved by home inhaler", "en"),
        ("Severe flank pain suspected renal colic", "en")
    ]

    l4_complaints = [
        ("Mild fever, runny nose and sore throat for 2 days", "en"),
        ("हल्का बुखार और खांसी जुकाम", "hi"),
        ("తేలికపాటి జ్వరం మరియు దగ్గు జలుబు", "te"),
        ("Sprained left wrist while lifting heavy bucket", "en"),
        ("Itchy red rash on both forearms after insect contact", "en"),
        ("Chronic dull lower back ache for 2 weeks", "en"),
        ("Mild non-bloody watery diarrhea 3 times today", "en")
    ]

    l5_complaints = [
        ("Post-operative suture removal on abdomen", "en"),
        ("टांके कटवाना और पट्टी बदलना", "hi"),
        ("కుట్లు తీయడం, కట్టు మార్చడం", "te"),
        ("Hypertension routine medication refill (Amlodipine)", "en"),
        ("General routine health checkup for job certificate", "en"),
        ("Dressing change for superficial healing abrasion", "en")
    ]

    for i in range(1, 801):
        enc_id = f"SYN-STRESS-{i:04d}"
        is_female = (random.random() < 0.48)
        first_name = random.choice(names_female if is_female else names_male)
        last_name = random.choice(surnames)
        gender = "Female" if is_female else "Male"
        age = random.randint(1, 85)

        # Arrival spread over last 480 minutes (8 hour OPD shift)
        mins_ago = random.randint(5, 480)

        # Target distribution
        dice = random.random()
        
        # 3% Abstention condition
        if dice < 0.03:
            raw = {
                "encounter_id": enc_id,
                "patient_name": f"[SYNTHETIC] {first_name} {last_name}",
                "age": age,
                "gender": gender,
                "spo2": None,
                "bp_systolic": None,
                "bp_diastolic": None,
                "pulse": None,
                "temperature": None,
                "complaint_original": "",
                "known_conditions": None,
                "language_hint": "en"
            }
        elif dice < 0.08:  # 5% L1
            comp, lang = random.choice(l1_complaints)
            raw = {
                "encounter_id": enc_id,
                "patient_name": f"[SYNTHETIC] {first_name} {last_name}",
                "age": age,
                "gender": gender,
                "spo2": random.choice([78.0, 81.0, 84.0, 88.0]),
                "bp_systolic": random.choice([60, 65, 80, 140]),
                "bp_diastolic": 40,
                "pulse": random.choice([165, 172, 32, 145]),
                "temperature": random.choice([98.6, 106.2, 99.0]),
                "complaint_original": comp,
                "known_conditions": "Critical",
                "language_hint": lang
            }
        elif dice < 0.23:  # 15% L2
            comp, lang = random.choice(l2_complaints)
            raw = {
                "encounter_id": enc_id,
                "patient_name": f"[SYNTHETIC] {first_name} {last_name}",
                "age": age,
                "gender": gender,
                "spo2": random.choice([89.0, 91.0, 95.0, 96.0]),
                "bp_systolic": random.choice([85, 185, 190, 130]),
                "bp_diastolic": random.choice([55, 122, 85]),
                "pulse": random.choice([135, 138, 42, 98]),
                "temperature": random.choice([98.6, 103.8, 99.2]),
                "complaint_original": comp,
                "known_conditions": "Hypertension/Cardiac",
                "language_hint": lang
            }
        elif dice < 0.58:  # 35% L3
            comp, lang = random.choice(l3_complaints)
            raw = {
                "encounter_id": enc_id,
                "patient_name": f"[SYNTHETIC] {first_name} {last_name}",
                "age": age,
                "gender": gender,
                "spo2": random.choice([94.0, 95.0, 96.0]),
                "bp_systolic": random.choice([115, 125, 135]),
                "bp_diastolic": random.choice([75, 80, 85]),
                "pulse": random.choice([102, 108, 112]),
                "temperature": random.choice([101.5, 102.2, 99.5]),
                "complaint_original": comp,
                "known_conditions": None,
                "language_hint": lang
            }
        elif dice < 0.88:  # 30% L4
            comp, lang = random.choice(l4_complaints)
            raw = {
                "encounter_id": enc_id,
                "patient_name": f"[SYNTHETIC] {first_name} {last_name}",
                "age": age,
                "gender": gender,
                "spo2": random.choice([97.0, 98.0, 99.0]),
                "bp_systolic": random.choice([110, 120, 128]),
                "bp_diastolic": random.choice([70, 78, 82]),
                "pulse": random.choice([72, 78, 84]),
                "temperature": random.choice([98.4, 99.1, 99.6]),
                "complaint_original": comp,
                "known_conditions": None,
                "language_hint": lang
            }
        else:  # 12% L5
            comp, lang = random.choice(l5_complaints)
            raw = {
                "encounter_id": enc_id,
                "patient_name": f"[SYNTHETIC] {first_name} {last_name}",
                "age": age,
                "gender": gender,
                "spo2": 99.0,
                "bp_systolic": 118,
                "bp_diastolic": 76,
                "pulse": 72,
                "temperature": 98.4,
                "complaint_original": comp,
                "known_conditions": "Stable",
                "language_hint": lang
            }

        # Status distribution: older arrivals may be completed
        status = "Waiting"
        if mins_ago > 300 and random.random() < 0.6:
            status = "Completed"
        elif mins_ago > 180 and random.random() < 0.2:
            status = "In Consultation"

        record = create_encounter_record(raw, minutes_ago=mins_ago, status=status)
        records.append(record)

    return records


# Labeled benchmark set for Evaluation Page (Real computed metrics)
EVALUATION_GROUND_TRUTH_DATASET: List[Dict[str, Any]] = [
    # True Level 1 Emergency
    {"complaint": "Patient collapsed, gasping agonal breaths, unresponsive", "spo2": 80.0, "bp_systolic": 60, "bp_diastolic": 40, "pulse": 170, "temp": 98.6, "ground_truth_tier": 1, "is_critical_emergency": True},
    {"complaint": "సాంస్ బంద్, పల్స్ లేదు (Not breathing, pulseless)", "spo2": 75.0, "bp_systolic": 0, "bp_diastolic": 0, "pulse": 25, "temp": 97.0, "ground_truth_tier": 1, "is_critical_emergency": True},
    {"complaint": "Active status epilepticus seizure lasting 15 mins", "spo2": 90.0, "bp_systolic": 130, "bp_diastolic": 80, "pulse": 140, "temp": 100.2, "ground_truth_tier": 1, "is_critical_emergency": True},
    {"complaint": "Severe hyperpyrexia heat stroke crisis", "spo2": 94.0, "bp_systolic": 110, "bp_diastolic": 70, "pulse": 150, "temp": 106.5, "ground_truth_tier": 1, "is_critical_emergency": True},
    # True Level 2 Emergency
    {"complaint": "Crushing retrosternal chest pain with diaphoresis", "spo2": 97.0, "bp_systolic": 140, "bp_diastolic": 90, "pulse": 95, "temp": 98.6, "ground_truth_tier": 2, "is_critical_emergency": True},
    {"complaint": "छाती में दर्द और सांस फूलना (Chest pain and dyspnea)", "spo2": 90.0, "bp_systolic": 110, "bp_diastolic": 70, "pulse": 105, "temp": 99.0, "ground_truth_tier": 2, "is_critical_emergency": True},
    {"complaint": "Acute stroke symptoms, right side paralysis and facial droop", "spo2": 96.0, "bp_systolic": 170, "bp_diastolic": 100, "pulse": 88, "temp": 98.4, "ground_truth_tier": 2, "is_critical_emergency": True},
    {"complaint": "Ingested agricultural pesticide poison 1 hour ago", "spo2": 95.0, "bp_systolic": 100, "bp_diastolic": 65, "pulse": 120, "temp": 98.6, "ground_truth_tier": 2, "is_critical_emergency": True},
    {"complaint": "పాము కాటు, తీవ్రమైన నొప్పి (Snake bite)", "spo2": 98.0, "bp_systolic": 115, "bp_diastolic": 75, "pulse": 115, "temp": 98.6, "ground_truth_tier": 2, "is_critical_emergency": True},
    {"complaint": "Silent hypoxia post viral illness", "spo2": 88.0, "bp_systolic": 120, "bp_diastolic": 80, "pulse": 90, "temp": 98.8, "ground_truth_tier": 2, "is_critical_emergency": True},
    {"complaint": "Hypertensive crisis with blinding headache", "spo2": 98.0, "bp_systolic": 200, "bp_diastolic": 125, "pulse": 92, "temp": 98.6, "ground_truth_tier": 2, "is_critical_emergency": True},
    # True Level 3 Urgent
    {"complaint": "High fever 102F and severe persistent vomiting", "spo2": 96.0, "bp_systolic": 110, "bp_diastolic": 70, "pulse": 104, "temp": 102.4, "ground_truth_tier": 3, "is_critical_emergency": False},
    {"complaint": "तेज़ बुखार और ठंड लगना (High fever with chills)", "spo2": 97.0, "bp_systolic": 115, "bp_diastolic": 75, "pulse": 100, "temp": 102.0, "ground_truth_tier": 3, "is_critical_emergency": False},
    {"complaint": "Deformed wrist after fall, suspected bone fracture", "spo2": 99.0, "bp_systolic": 130, "bp_diastolic": 80, "pulse": 82, "temp": 98.4, "ground_truth_tier": 3, "is_critical_emergency": False},
    {"complaint": "Severe colic flank pain radiating to groin", "spo2": 98.0, "bp_systolic": 135, "bp_diastolic": 85, "pulse": 96, "temp": 98.6, "ground_truth_tier": 3, "is_critical_emergency": False},
    # True Level 4 Less Urgent
    {"complaint": "Mild cough and common cold symptoms for 3 days", "spo2": 99.0, "bp_systolic": 120, "bp_diastolic": 80, "pulse": 74, "temp": 98.6, "ground_truth_tier": 4, "is_critical_emergency": False},
    {"complaint": "గజ్జల్లో దురద, తేలికపాటి జ్వరం (Itch and mild fever)", "spo2": 98.0, "bp_systolic": 118, "bp_diastolic": 76, "pulse": 76, "temp": 99.1, "ground_truth_tier": 4, "is_critical_emergency": False},
    {"complaint": "Sprained ankle while walking, mild swelling", "spo2": 98.0, "bp_systolic": 122, "bp_diastolic": 78, "pulse": 80, "temp": 98.4, "ground_truth_tier": 4, "is_critical_emergency": False},
    # True Level 5 Non-Urgent
    {"complaint": "Routine blood pressure medication refill", "spo2": 98.0, "bp_systolic": 120, "bp_diastolic": 80, "pulse": 70, "temp": 98.4, "ground_truth_tier": 5, "is_critical_emergency": False},
    {"complaint": "Suture removal and sterile dressing change", "spo2": 99.0, "bp_systolic": 116, "bp_diastolic": 74, "pulse": 72, "temp": 98.2, "ground_truth_tier": 5, "is_critical_emergency": False},
    # Safety Abstentions
    {"complaint": "", "spo2": None, "bp_systolic": None, "bp_diastolic": None, "pulse": None, "temp": None, "ground_truth_tier": None, "should_abstain": True, "is_critical_emergency": False},
    {"complaint": "asdf", "spo2": None, "bp_systolic": None, "bp_diastolic": None, "pulse": None, "temp": None, "ground_truth_tier": None, "should_abstain": True, "is_critical_emergency": False}
]
