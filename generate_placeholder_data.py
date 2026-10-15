"""CarePriority Gov - Synthetic Placeholder Dataset Generator
============================================================
Creates a clearly labeled synthetic placeholder dataset in /data/synthetic_placeholder.csv
when organizer files (~30k synthetic intake records) have not yet been downloaded.

Columns match the organizer specification:
encounter_id, age, sex, bp_systolic, bp_diastolic, pulse, temperature, spo2,
chief_complaint, known_conditions, medications, true_level, true_department, true_emergency (~4%).
"""

import os
import random
from typing import Optional
import pandas as pd

def generate_placeholder_dataset(output_path: Optional[str] = "data/synthetic_placeholder.csv", n_samples: int = 600):
    if output_path:
        try:
            dir_path = os.path.dirname(output_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
        except Exception:
            pass
    random.seed(42)

    departments = [
        "General Medicine", "Cardiology", "Pediatrics", "Orthopedics",
        "Emergency / Resuscitation", "Obstetrics & Gynecology", "ENT", "Respiratory / Pulmonology"
    ]

    # Red-flag & emergency complaints (English, Hindi, Telugu, Hinglish, Telugu-Latin)
    l1_complaints = [
        "Sudden collapse at OPD gate, gasping, turning blue",
        "Cardiac arrest in reception, unresponsive, agonal breathing",
        "saans nahi aa rahi, behosh ho gaya hai",
        "oopiri aadatledu, spruha ledu, neelamga maaripoyindi",
        "Active status epilepticus seizure, uncal herniation suspected",
        "Massive projectile hematemesis with profound shock"
    ]

    l2_complaints = [
        "Crushing retrosternal chest pain radiating to left jaw and diaphoresis",
        "seene me dard, chaati me bahut teevra dard aur paseena",
        "gunde noppi, chaati noppi, edama cheyyi laavutondi",
        "Acute ischemic stroke, right hemiplegia and facial droop since 45 mins",
        "Accidental organophosphate pesticide ingestion, pinpoint pupils",
        "paamu kaatu on right ankle with progressive edema and vomiting",
        "Silent hypoxia with severe fatigue and SpO2 86%",
        "Hypertensive encephalopathy with systolic BP 210 and altered sensorium",
        "Heavy vaginal bleeding in 3rd trimester pregnancy with hypotension"
    ]

    l3_complaints = [
        "High grade fever 102.5F with chills and continuous vomiting for 3 days",
        "tez bukhar aur pet me tez dard",
        "teevra jwaram, vanti, kadupu noppi",
        "Deformed wrist after falling from two wheeler, unable to move fingers",
        "Severe acute left flank pain radiating to groin, hematuria",
        "Exacerbation of known bronchial asthma, audible wheezing",
        "Deep contaminated laceration on foot from rusty nail, throbbing pain"
    ]

    l4_complaints = [
        "Mild fever, runny nose and throat irritation for 2 days",
        "halka bukhar aur khansi jukam",
        "telikapati jwaram, daggu jalubu",
        "Twisted right ankle while stepping down bus, mild swelling",
        "Generalized itchy urticarial skin rash after eating seafood",
        "Chronic lower backache radiating to buttocks for 3 weeks",
        "Dysuria and burning sensation during urination without fever"
    ]

    l5_complaints = [
        "Hypertension routine medication refill (Amlodipine 5mg)",
        "dawa likhwana, regular BP checkup",
        "Post-operative suture removal on abdomen, surgical wound clean",
        "patti badalna, dressing change for superficial healing scratch",
        "General health fitness certificate for employment",
        "Annual routine diabetic blood sugar test follow-up"
    ]

    rows = []
    for i in range(1, n_samples + 1):
        enc_id = f"SYN-PLACEHOLDER-{i:04d}"
        dice = random.random()
        sex = random.choice(["Male", "Female"])
        age = random.randint(2, 85)

        # ~4% True Emergency flag distribution as specified by organizer
        # Level distribution: L1 ~ 2%, L2 ~ 8%, L3 ~ 35%, L4 ~ 35%, L5 ~ 20%
        if dice < 0.02:  # L1
            true_level = 1
            true_emergency = 1
            true_dept = "Emergency / Resuscitation"
            comp = random.choice(l1_complaints)
            spo2 = round(random.uniform(74.0, 84.5), 1)
            pulse = random.choice([32, 165, 175])
            bp_sys = random.choice([0, 60, 68, 75])  # Note: 0 tested
            bp_dia = random.choice([0, 40, 50])
            temp = random.choice([98.4, 106.2])
            conds = random.choice(["COPD", "Severe CAD", "Epilepsy", None])
            meds = random.choice(["Inhalers", "Aspirin", "Phenytoin", None])

        elif dice < 0.10:  # L2 (~8%, total L1+L2 ~10%, of which ~4% absolute are true critical emergencies)
            true_level = 2
            true_emergency = 1 if dice < 0.04 else 0
            comp = random.choice(l2_complaints)
            if "chest" in comp.lower() or "seene" in comp.lower() or "gunde" in comp.lower():
                true_dept = "Cardiology"
            elif "stroke" in comp.lower() or "paralysis" in comp.lower():
                true_dept = "General Medicine"
            elif "paamu" in comp.lower() or "pesticide" in comp.lower():
                true_dept = "Emergency / Resuscitation"
            elif "pregnancy" in comp.lower():
                true_dept = "Obstetrics & Gynecology"
                sex = "Female"
            else:
                true_dept = random.choice(["Cardiology", "Respiratory / Pulmonology", "Emergency / Resuscitation"])

            spo2 = round(random.uniform(88.0, 93.0), 1)
            pulse = random.choice([132, 138, 42, 104])
            bp_sys = random.choice([85, 185, 195, 130])
            bp_dia = random.choice([55, 122, 85])
            temp = random.choice([98.6, 103.8])
            conds = random.choice(["Hypertension", "Diabetes", "Asthma", None])
            meds = random.choice(["Amlodipine", "Metformin", "Salbutamol", None])

        elif dice < 0.45:  # L3
            true_level = 3
            true_emergency = 0
            comp = random.choice(l3_complaints)
            if "fracture" in comp.lower() or "wrist" in comp.lower():
                true_dept = "Orthopedics"
            elif "asthma" in comp.lower():
                true_dept = "Respiratory / Pulmonology"
            elif age <= 14:
                true_dept = "Pediatrics"
            else:
                true_dept = "General Medicine"

            spo2 = round(random.uniform(94.0, 97.0), 1)
            pulse = random.randint(88, 114)
            bp_sys = random.randint(110, 138)
            bp_dia = random.randint(70, 88)
            temp = round(random.uniform(100.5, 103.0), 1)
            conds = random.choice(["Asthma", "Kidney stones", None])
            meds = random.choice(["Paracetamol", "Inhaler", None])

        elif dice < 0.80:  # L4
            true_level = 4
            true_emergency = 0
            comp = random.choice(l4_complaints)
            if "ankle" in comp.lower() or "backache" in comp.lower():
                true_dept = "Orthopedics"
            elif "throat" in comp.lower() or "ear" in comp.lower():
                true_dept = "ENT"
            elif age <= 14:
                true_dept = "Pediatrics"
            else:
                true_dept = "General Medicine"

            spo2 = round(random.uniform(97.0, 99.5), 1)
            pulse = random.randint(68, 86)
            bp_sys = random.randint(112, 126)
            bp_dia = random.randint(72, 84)
            temp = round(random.uniform(98.2, 99.4), 1)
            conds = random.choice(["Seasonal allergy", None])
            meds = random.choice(["Cetirizine", None])

        else:  # L5
            true_level = 5
            true_emergency = 0
            comp = random.choice(l5_complaints)
            true_dept = "General Medicine"
            spo2 = 99.0
            pulse = random.randint(65, 80)
            bp_sys = random.randint(115, 128)
            bp_dia = random.randint(74, 84)
            temp = 98.4
            conds = random.choice(["Hypertension", "Type 2 Diabetes", None])
            meds = random.choice(["Amlodipine", "Metformin", "Vitamins", None])

        # Random missing values (~5% missing vitals to simulate real intake)
        if random.random() < 0.05:
            spo2 = None
        if random.random() < 0.05:
            bp_sys = None
            bp_dia = None
        if random.random() < 0.05:
            pulse = None

        rows.append({
            "encounter_id": enc_id,
            "age": age,
            "sex": sex,
            "bp_systolic": bp_sys,
            "bp_diastolic": bp_dia,
            "pulse": pulse,
            "temperature": temp,
            "spo2": spo2,
            "chief_complaint": comp,
            "known_conditions": conds,
            "medications": meds,
            "true_level": true_level,
            "true_department": true_dept,
            "true_emergency": true_emergency,
            "is_synthetic_placeholder": True
        })

    df = pd.DataFrame(rows)
    if output_path:
        try:
            df.to_csv(output_path, index=False)
            print(f"Generated {len(df)} synthetic placeholder records at {output_path}")
            print(f"True emergency prevalence: {df['true_emergency'].mean()*100:.1f}% (~4% target)")
        except Exception:
            pass
    return df

if __name__ == "__main__":
    generate_placeholder_dataset()
