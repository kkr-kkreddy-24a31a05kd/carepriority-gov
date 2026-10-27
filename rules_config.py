"""CarePriority Gov - Clinical Triage Rules Configuration
======================================================
Demo values — must be reviewed by a clinician before any real use.
This configuration file centralizes all clinical rules, physiological thresholds,
red-flag symptom dictionaries (English, Hindi, Telugu, Hinglish, Telugu-in-Latin),
department mappings, abstention triggers, asymmetric 20x cost matrix,
and hospital capacity settings.

DISCLAIMER:
Demo scale inspired by 5-level triage systems (such as ESI / MTS) —
not a validated ESI implementation. Independent demonstration project,
not an official government service.
"""

from typing import Dict, List, Any, Optional

# ==============================================================================
# 1. TRIAGE LEVEL DEFINITIONS & COLOR CODING
# ==============================================================================
# Demo values — must be reviewed by a clinician before any real use.
TRIAGE_LEVELS = {
    1: {
        "name": "Level 1: Resuscitation",
        "short_name": "L1 - Resuscitation",
        "target_time_mins": 0,
        "badge_color": "#FF2A55",
        "text_color": "#FFFFFF",
        "pulse_glow": True,
        "is_emergency": True,
        "description": "Immediate life-saving intervention required without delay."
    },
    2: {
        "name": "Level 2: Emergent",
        "short_name": "L2 - Emergent",
        "target_time_mins": 15,
        "badge_color": "#FF7A00",
        "text_color": "#FFFFFF",
        "pulse_glow": False,
        "is_emergency": True,
        "description": "High risk situation, severe distress, or unstable vital signs."
    },
    3: {
        "name": "Level 3: Urgent",
        "short_name": "L3 - Urgent",
        "target_time_mins": 45,
        "badge_color": "#FFC700",
        "text_color": "#1A1A1A",
        "pulse_glow": False,
        "is_emergency": False,
        "description": "Moderate distress or condition requiring multiple diagnostic resources."
    },
    4: {
        "name": "Level 4: Less Urgent",
        "short_name": "L4 - Less Urgent",
        "target_time_mins": 90,
        "badge_color": "#00A3FF",
        "text_color": "#FFFFFF",
        "pulse_glow": False,
        "is_emergency": False,
        "description": "Stable condition requiring simple investigation or single intervention."
    },
    5: {
        "name": "Level 5: Non-Urgent",
        "short_name": "L5 - Non-Urgent",
        "target_time_mins": 180,
        "badge_color": "#00D68F",
        "text_color": "#081528",
        "pulse_glow": False,
        "is_emergency": False,
        "description": "Routine minor complaint or medication refill; low acute risk."
    }
}

# ==============================================================================
# 2. VITAL SIGNS VALIDATION & PLAUSIBILITY BOUNDS
# ==============================================================================
# Demo values — must be reviewed by a clinician before any real use.
# DATA ENTRY RULES:
# - BP of 0 = treated as missing, flagged for verification, stored as None.
# - Temperature: if unit is Celsius and value is above 43 (e.g. 96 C), flag as
#   probable unit error (likely Fahrenheit). Never silently trust it. If unit is
#   Fahrenheit and value is outside a plausible human range (90-110F), also flag it.
# - Missing values stay None, never 0.
VITALS_PLAUSIBILITY = {
    "age": {
        "min": 0,
        "max": 130,
        "error_msg": "Age must be a valid non-negative number between 0 and 130."
    },
    "spo2": {
        "min": 0,
        "max": 100,
        "error_msg": "SpO2 must be between 0% and 100%."
    },
    "bp_systolic": {
        "min": 40,
        "max": 300,
        "zero_warning": "Systolic BP of 0 detected: Treated as missing/unobtainable and stored as None; flagged for verification.",
        "error_msg": "Systolic BP must be between 40 and 300 mmHg (BP of 0 treated as missing)."
    },
    "bp_diastolic": {
        "min": 20,
        "max": 200,
        "error_msg": "Diastolic BP must be between 20 and 200 mmHg."
    },
    "pulse": {
        "min": 20,
        "max": 300,
        "error_msg": "Pulse rate must be between 20 and 300 bpm."
    },
    "temperature": {
        "min_fahrenheit": 90.0,
        "max_fahrenheit": 110.0,
        "celsius_max_plausible": 43.0,
        "probable_celsius_low": 34.0,
        "probable_celsius_high": 43.0,
        "celsius_unit_error_msg": "Temperature entered in Celsius above 43°C (e.g., 96°C) flagged as a probable unit error (likely intended as Fahrenheit). Never silently trusted.",
        "fahrenheit_error_msg": "Temperature outside plausible human range (90.0°F - 110.0°F). Flagged as a probable unit/entry error rather than silently accepted."
    }
}

# ==============================================================================
# 3. MANDATORY HARD-CODED VITAL ESCALATION THRESHOLDS
# ==============================================================================
# Demo values — must be reviewed by a clinician before any real use.
# These rules force an escalation to Level 1 or Level 2 and CANNOT be downgraded by scoring.
HARD_ESCALATION_VITALS = {
    # Extreme Critical (Forces Level 1: Resuscitation)
    "L1": {
        "spo2_critical": 85.0,        # SpO2 < 85% -> L1
        "pulse_arrest_high": 160,     # Pulse > 160 bpm in adults -> L1
        "pulse_arrest_low": 35,       # Pulse < 35 bpm -> L1
        "bp_systolic_shock": 70,      # Systolic BP < 70 mmHg -> L1 profound shock
        "temp_hyperpyrexia": 106.0,   # Temp >= 106°F -> L1 heatstroke/severe neuro crisis
    },
    # Severe Urgent (Forces at least Level 2: Emergent)
    "L2": {
        "spo2_hypoxia": 92.0,         # SpO2 < 92% -> mandatory L2 (or L1 if < 85)
        "bp_systolic_hypotension": 90,# Systolic BP < 90 mmHg -> L2
        "bp_systolic_crisis": 180,    # Systolic BP >= 180 mmHg -> L2 hypertensive crisis
        "bp_diastolic_crisis": 120,   # Diastolic BP >= 120 mmHg -> L2 hypertensive crisis
        "pulse_tachycardia": 130,     # Pulse >= 130 bpm -> L2
        "pulse_bradycardia": 45,      # Pulse <= 45 bpm -> L2
        "temp_high_fever": 103.5,     # Temp >= 103.5°F -> L2
        "temp_hypothermia": 94.0      # Temp < 94.0°F -> L2 hypothermia
    }
}

# ==============================================================================
# 4. RED-FLAG SYMPTOM KEYWORDS (ENGLISH, HINDI, TELUGU, HINGLISH, TELUGU-IN-LATIN)
# ==============================================================================
# Demo values — must be reviewed by a clinician before any real use.
# Any match triggers mandatory escalation to Level 1 or Level 2.
RED_FLAG_KEYWORDS = {
    "L1_SYMPTOMS": {
        "rule_name": "Resuscitation Keyword Detected",
        "tier": 1,
        "keywords": [
            # English
            "cardiac arrest", "not breathing", "unresponsive", "unconscious", 
            "respiratory arrest", "severe anaphylaxis", "cyanosis", "turning blue",
            "active seizure", "status epilepticus", "massive hemorrhage", "collapsed",
            "agonal breathing", "stridor", "asphyxiation", "gasping", "pulseless",
            # Hindi (Devanagari)
            "सांस नहीं आ रही", "सांस बंद", "बेहोश", "बेहोशी", "अचेत", 
            "दिल रुक गया", "नीला पड़ गया", "लगातार दौरा", "सांस रुकना",
            # Telugu (Telugu script)
            "ఊపిరాడటం లేదు", "శ్వాస ఆడటం లేదు", "స్పృహ లేదు", "తెలివి తప్పింది",
            "గుండె ఆగిపోయింది", "నీలంగా మారిపోయింది", "సృహ కోల్పోయాడు", "శ్వాస నిలిచిపోయింది",
            # Hinglish (Transliterated Hindi in Latin script)
            "saans nahi aa rahi", "sans nahi aa rahi", "saans band", "sans band",
            "behoshi", "behosh", "dil ruk gaya", "neela pad gaya", "saans rukna",
            "hosh nahi hai", "chetna nahi hai",
            # Telugu-in-Latin-script
            "oopiri aadatledu", "upiri aadatledu", "swasa aadatledu", "svasa aadatledu",
            "spruha ledu", "telivi tappindi", "gunde aagipoyindi", "neelamga maaripoyindi",
            "spruha kolpoyadu", "swasa nilichipoyindi", "sohi ledu"
        ]
    },
    "L2_SYMPTOMS": {
        "rule_name": "Emergent Red-Flag Keyword Detected",
        "tier": 2,
        "keywords": [
            # English
            "chest pain", "heart attack", "angina", "shortness of breath", 
            "dyspnea", "severe bleeding", "profuse bleeding", "stroke", 
            "paralysis", "facial droop", "slurred speech", "weakness in arm", 
            "poisoning", "toxic ingestion", "snake bite", "poison", "insecticide",
            "seizure", "fit", "fits", "convulsion", "suicidal", "severe burn",
            "severe head injury", "coughing blood", "hemoptysis", "vomiting blood",
            "hematemesis", "acute abdominal rigidity", "altered mental state",
            # Hindi (Devanagari)
            "छाती में दर्द", "सीने में दर्द", "दिल का दौरा", "छाती में जकड़न",
            "खून बह रहा", "भारी रक्तस्राव", "लकवा", "पक्षाघात", "मुंह टेढ़ा",
            "बोली लड़खड़ाना", "ज़हर", "विष", "सांप का काटना", "दौरा", "मिर्गी",
            "खून की उल्टी", "खांसी में खून", "सिर में गंभीर चोट", "पेट में भयंकर दर्द",
            # Telugu (Telugu script)
            "గుండె నొప్పి", "ఛాతీ నొప్పి", "ఛాతీలో మంట", "తీవ్ర రక్తస్రావం",
            "రక్తం కారుతోంది", "పక్షవాతం", "ముఖం వంకర", "మాట తడబడటం",
            "విషం", "పాము కాటు", "పురుగుల మందు", "మూర్ఛ", "ఫిట్స్",
            "రక్తం వాంతి", "దగ్గులో రక్తం", "తలకి తీవ్ర గాయం", "కడుపులో భరించలేని నొప్పి",
            # Hinglish (Transliterated Hindi in Latin script)
            "seene me dard", "seene mein dard", "chaati me dard", "chaati mein dard",
            "chati me dard", "chati mein dard", "chhati me dard", "dil ka daura",
            "dil me dard", "khoon beh raha", "bhari raktasrav", "lakwa", "pakshaghat",
            "muh tedha", "boli ladkhadana", "zeher", "zaher", "vish", "saanp ka katna",
            "saanp ne kata", "daura", "dora", "mirgi", "khoon ki ulti", "khansi me khoon",
            "pet me bhayankar dard", "sir me gambhir chot", "saans lene me takleef",
            "sans phoolna", "saans phool rahi hai",
            # Telugu-in-Latin-script
            "gunde noppi", "chhati noppi", "chaati noppi", "chathi noppi",
            "chhati lo manta", "teevra raktasravam", "raktham karutondi",
            "raktham vastundi", "pakshavatham", "mukham vankara", "mata tadabadatam",
            "visham", "paamu kaatu", "purugula mandu", "moorcha",
            "raktham vaanti", "daggu lo raktham", "thala ki teevra gaayam",
            "kadupulo bharinchaleni noppi", "aayasam ekkuva ayindi", "gunde baadha"
        ]
    }
}

# ==============================================================================
# 5. SECONDARY SYMPTOM SCORING & URGENCY KEYWORDS
# ==============================================================================
# Demo values — must be reviewed by a clinician before any real use.
# Used for Level 3, Level 4, Level 5 differentiation when hard rules do not trigger.
SECONDARY_KEYWORDS = {
    "L3_URGENT": [
        # Moderate risk complaints
        "moderate fever", "persistent vomiting", "dehydration", "asthma flare",
        "deep laceration", "fracture suspected", "severe abdominal pain", 
        "kidney stone", "urinary retention", "dislocated", "wound infection",
        "high bp", "dizziness", "vertigo", "diabetic high sugar",
        # Hindi
        "तेज़ बुखार", "लगातार उल्टी", "हड्डी टूटना", "पेट दर्द", "चक्कर आना", "संक्रमण",
        # Telugu
        "తీవ్ర జ్వరం", "ఎడతెగని వాంతులు", "ఎముక విరిగింది", "కడుపు నొప్పి", "తలతిరగడం", "ఇన్ఫెక్షన్",
        # Hinglish & Telugu-Latin
        "tez bukhar", "tez bukhaar", "haddi tootna", "chakkar aana", "pet dard",
        "teevra jwaram", "emuka virigindi", "kadupu noppi", "talatiragadam"
    ],
    "L4_LESS_URGENT": [
        # Mild/stable conditions
        "mild fever", "cough and cold", "sore throat", "mild diarrhea",
        "earache", "minor burn", "sprain", "rash", "skin allergy",
        "back pain", "joint pain", "headache", "urinary burn", "dysuria",
        # Hindi
        "हल्का बुखार", "खांसी जुकाम", "गले में दर्द", "हल्के दस्त", "खुजली", "बदन दर्द", "सिरदर्द",
        # Telugu
        "తేలికపాటి జ్వరం", "దగ్గు జలుబు", "గొంతు నొప్పి", "విరేచనాలు", "దురద", "ఒళ్ళు నొప్పులు", "తలనెప్పి",
        # Hinglish & Telugu-Latin
        "halka bukhar", "khansi jukam", "gale me dard", "badan dard", "sirdard",
        "telikapati jwaram", "daggu jalubu", "gontu noppi", "talanappi"
    ],
    "L5_NON_URGENT": [
        # Routine, refills, minor checks
        "dressing change", "suture removal", "medication refill", "routine checkup",
        "health certificate", "vaccination", "followup", "mild itch", "normal report",
        # Hindi
        "दवा लिखवाना", "पट्टी बदलना", "टांके कटवाना", "नियमित जांच", "टीकाकरण",
        # Telugu
        "మందులు రాసుకోవడం", "కట్టు మార్చడం", "కుట్లు తీయడం", "సాధారణ పరీక్ష", "టీకా",
        # Hinglish & Telugu-Latin
        "dawa likhwana", "patti badalna", "tanke katwana", "mandulu raasukovadam"
    ]
}

# ==============================================================================
# 6. DEPARTMENT ROUTING RULES
# ==============================================================================
# Demo values — must be reviewed by a clinician before any real use.
# Exact 13 canonical departments matching organizer evaluator and trained ML model
DEPARTMENTS = [
    "General Medicine",
    "Paediatrics",
    "Obstetrics & Gynaecology",
    "General Surgery",
    "Orthopaedics",
    "ENT",
    "Ophthalmology",
    "Dermatology",
    "Psychiatry",
    "Dental",
    "Cardiology",
    "Pulmonology/Chest & TB",
    "Emergency/Casualty"
]

DEPARTMENT_RULES = {
    "Emergency/Casualty": {
        "force_on_tiers": [1],
        "keywords": [
            "trauma", "unconscious", "shock", "massive bleeding", "poisoning", 
            "snake bite", "cardiac arrest", "stridor", "accident", "gunshot",
            "दुर्घटना", "ज़हर", "सांप", "ప్రమాదం", "విషం",
            "durghatna", "zeher", "saanp", "pramadam", "visham"
        ]
    },
    "Cardiology": {
        "keywords": [
            "chest pain", "angina", "palpitation", "heart", "arrhythmia", 
            "cardiac", "hypertension", "bp high", "heart attack",
            "छाती", "दिल", "हार्ट", "గుండె", "ఛాతీ",
            "seene me dard", "chati me dard", "gunde noppi", "chaati noppi"
        ]
    },
    "Paediatrics": {
        "max_age": 14,
        "keywords": [
            "child", "baby", "infant", "newborn", "pediatric", "paediatric",
            "बच्चा", "शिशु", "बालक", "పిల్లవాడు", "శిశువు", "పాప", "బాబు",
            "bachha", "bacha", "shishu", "pillavadu", "paapa", "baabu"
        ]
    },
    "Pulmonology/Chest & TB": {
        "keywords": [
            "asthma", "breathing", "wheezing", "cough", "phlegm", "tuberculosis",
            "tb", "pneumonia", "shortness of breath", "copd", "stridor", "inhaler",
            "सांस", "दमा", "खांसी", "टीबी", "శ్వాస", "దగ్గు", "ఆయాసం",
            "saans", "dama", "khansi", "swasa", "daggu", "aayasam", "oopiri"
        ]
    },
    "Orthopaedics": {
        "keywords": [
            "bone", "fracture", "joint", "dislocation", "sprain", "knee", 
            "back pain", "spine", "fall", "injury to leg", "injury to arm",
            "हड्डी", "फ्रैक्चर", "जोड़", "कमर दर्द", "మోకాలు", "ఎముక", "కీళ్ల నొప్పులు",
            "haddi", "kamar dard", "jod", "emuka", "mokalu"
        ]
    },
    "Obstetrics & Gynaecology": {
        "gender": "Female",
        "keywords": [
            "pregnancy", "pregnant", "labor pain", "labor", "vaginal bleeding",
            "menstrual", "period cramps", "maternity", "antenatal", "fetus",
            "गर्भावस्था", "गर्भवती", "प्रसव", "గర్భం", "నెలసరి", "కాన్పు",
            "garbhavastha", "garbham", "prasavam", "pains"
        ]
    },
    "ENT": {
        "keywords": [
            "ear", "earache", "nose", "epistaxis", "nosebleed", "throat", 
            "tonsil", "hearing", "sinus", "hoarseness",
            "कान", "नाक", "गला", "टॉन्सिल", "చెవి", "ముక్కు", "గొంతు",
            "kaan", "naak", "gala", "chevi", "mukku", "gontu"
        ]
    },
    "Dental": {
        "keywords": [
            "dental", "tooth", "toothache", "teeth", "gum", "gum bleeding",
            "cavity", "jaw swelling", "wisdom tooth", "molar",
            "दांत", "मसूड़े", "పన్ను", "దంత", "dant", "danta"
        ]
    },
    "Dermatology": {
        "keywords": [
            "skin", "rash", "itching", "scabies", "ringworm", "eczema",
            "acne", "psoriasis", "hives", "dermatitis", "white patches",
            "त्वचा", "खुजली", "दाद", "చర్మం", "దురద", "khujli", "chaala"
        ]
    },
    "Ophthalmology": {
        "keywords": [
            "eye", "vision", "cataract", "red eye", "glasses", "cornea",
            "blurry vision", "blindness", "conjunctivitis", "foreign body in eye",
            "आंख", "दृष्टि", "मोतियाबिंद", "కన్ను", "చూపు", "aankh", "kannu"
        ]
    },
    "Psychiatry": {
        "keywords": [
            "depression", "anxiety", "low mood", "hallucination", "suicidal",
            "suicide", "sleep problem", "insomnia", "psychosis", "agitation",
            "alcohol withdrawal", "mental", "तनाव", "मानसिक", "నిద్రలేమి", "మానసిక"
        ]
    },
    "General Surgery": {
        "keywords": [
            "surgery", "cut", "wound", "boil", "abscess", "cellulitis",
            "diabetic foot", "dog bite", "bite", "appendicitis", "kidney stone",
            "gallbladder", "gall bladder", "hernia", "piles", "fistula", "breast lump",
            "gangrene", "घाव", "चोट", "ऑपरेशन", "గాయం", "కుట్లు", "ghaav", "gaayam"
        ]
    },
    "General Medicine": {
        "is_default": True,
        "keywords": [
            "fever", "weakness", "vomiting", "diarrhea", "headache", 
            "fatigue", "body ache", "diabetes", "infection", "jaundice",
            "बुखार", "कमजोरी", "उल्टी", "दस्त", "జ్వరం", "నీరసం",
            "bukhar", "kamzori", "ulti", "dast", "jwaram", "neerasam"
        ]
    }
}

# Backward compatibility aliases for legacy callers
DEPARTMENT_RULES["Pediatrics"] = DEPARTMENT_RULES["Paediatrics"]
DEPARTMENT_RULES["Emergency / Resuscitation"] = DEPARTMENT_RULES["Emergency/Casualty"]
DEPARTMENT_RULES["Respiratory / Pulmonology"] = DEPARTMENT_RULES["Pulmonology/Chest & TB"]
DEPARTMENT_RULES["Orthopedics"] = DEPARTMENT_RULES["Orthopaedics"]
DEPARTMENT_RULES["Obstetrics & Gynecology"] = DEPARTMENT_RULES["Obstetrics & Gynaecology"]

# ==============================================================================
# 7. SAFETY ABSTENTION TRIGGER CONDITIONS
# ==============================================================================
# Demo values — must be reviewed by a clinician before any real use.
# If these conditions are met, system MUST abstains and escalate to Nurse Review.
ABSTENTION_CONDITIONS = {
    "missing_all_data": {
        "rule_name": "No Complaint and No Vitals Provided",
        "description": "Patient intake has neither a free-text complaint nor any vital signs. Cannot safely assign a priority."
    },
    "contradictory_vitals": {
        "rule_name": "Physiologically Contradictory Vitals",
        "description": "e.g., recorded Pulse is 0 or unreadable while patient is labeled fully ambulatory with normal BP."
    },
    "ambiguous_gibberish": {
        "rule_name": "Ambiguous or Uninterpretable Complaint with Incomplete Vitals",
        "description": "Complaint contains no recognizable clinical tokens and essential vitals (Pulse, SpO2, BP) are missing."
    },
    "low_model_confidence": {
        "rule_name": "Low Model Confidence with Incomplete Clinical Indicators",
        "threshold": 0.35,
        "description": "ML triage model confidence is below safety threshold with no clear vital signs or symptoms."
    }
}

# ==============================================================================
# 8. QUEUE & HOSPITAL CAPACITY DEFAULTS
# ==============================================================================
# Demo values — must be reviewed by a clinician before any real use.
HOSPITAL_CONFIG = {
    "hospital_name": "District Civil Hospital - OPD Command Center",
    "default_doctor_count": 2,
    "target_patients_per_doctor_hour": 15,
    "anti_starvation_threshold_mins": {
        4: 120,   # Flag Level 4 if waiting > 2 hours
        5: 180    # Flag Level 5 if waiting > 3 hours
    },
    "auto_refresh_interval_secs": 15
}

# ==============================================================================
# 9. ASYMMETRIC SAFETY COST MATRIX (20x UNDER-TRIAGE PENALTY)
# ==============================================================================
# Under-triaging a true emergency costs ~20x more than over-triaging a mild case.
# Never optimize plain accuracy!
# - False Negative (True emergency L1/L2 classified as non-emergency L3/L4/L5): Cost = 20.0
# - False Positive (Non-emergency L3/L4/L5 classified as emergency L1/L2): Cost = 1.0
# - Exact match: Cost = 0.0
ASYMMETRIC_COST_CONFIG = {
    "under_triage_emergency_cost": 20.0,
    "over_triage_cost": 1.0,
    "tier_under_triage_step_cost": 5.0,  # Each tier level under-triaged
    "tier_over_triage_step_cost": 1.0    # Each tier level over-triaged
}

# ==============================================================================
# Demo values - must be reviewed by a clinician before any real use.
# 5x5 Cost Matrix: [actual_tier - 1][predicted_tier - 1]
# Rows: Actual Ground Truth Tier (L1 to L5)
# Cols: Predicted Tier (L1 to L5)
# Notice the extreme 20x+ penalty when actual is L1/L2 and predicted is L3/L4/L5!
TRIAGE_COST_MATRIX_5X5 = [
    # Actual L1 (Resuscitation)
    [0.0,  2.0, 20.0, 40.0, 60.0],
    # Actual L2 (Emergent)
    [1.0,  0.0, 20.0, 30.0, 40.0],
    # Actual L3 (Urgent)
    [2.0,  1.0,  0.0,  5.0, 10.0],
    # Actual L4 (Less Urgent)
    [2.0,  1.0,  1.0,  0.0,  2.0],
    # Actual L5 (Non-Urgent)
    [2.0,  1.0,  1.0,  1.0,  0.0]
]

# Aliases for flexible import compatibility
COST_MATRIX = TRIAGE_COST_MATRIX_5X5
ASYMMETRIC_COST_MATRIX = TRIAGE_COST_MATRIX_5X5


def calculate_asymmetric_cost(actual_tier: int, predicted_tier: int) -> float:
    """
    Computes asymmetric clinical risk cost between actual and predicted triage tier.
    Returns penalty float based on TRIAGE_COST_MATRIX_5X5.
    """
    if not (1 <= actual_tier <= 5) or not (1 <= predicted_tier <= 5):
        return 0.0
    return TRIAGE_COST_MATRIX_5X5[actual_tier - 1][predicted_tier - 1]
