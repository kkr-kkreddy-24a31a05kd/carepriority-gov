"""CarePriority Gov - Database Layer & Session State Fallback
==========================================================
Unified database abstraction supporting:
1. Production MongoDB via PyMongo (indexes on status, tier, created_at)
2. Automatic seamless fallback to st.session_state when MongoDB is unreachable
   (with short serverSelectionTimeoutMS = 1500ms)
3. Zero API drift: identical interface for both modes.
"""

import os
import copy
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

load_dotenv()

# Check for streamlit import for session state caching
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# MongoDB driver
try:
    from pymongo import MongoClient, ASCENDING, DESCENDING
    from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False


def get_mongo_config() -> Tuple[str, str]:
    """
    Reads MongoDB configuration with strict precedence:
    1. st.secrets (Streamlit Community Cloud / .streamlit/secrets.toml)
    2. os.environ (Environment variables / .env)
    3. Fallback defaults (empty string for MONGO_URI -> direct session fallback; 'carepriority_gov' for DB_NAME)
    """
    mongo_uri = ""
    db_name = ""

    # 1. Read from st.secrets first
    if HAS_STREAMLIT:
        try:
            if hasattr(st, "secrets"):
                if "MONGO_URI" in st.secrets:
                    val = str(st.secrets["MONGO_URI"]).strip()
                    if val:
                        mongo_uri = val
                if "DB_NAME" in st.secrets:
                    val = str(st.secrets["DB_NAME"]).strip()
                    if val:
                        db_name = val
        except Exception:
            pass

    # 2. Read from environment variables second
    if not mongo_uri:
        mongo_uri = os.getenv("MONGO_URI", "").strip()
    if not db_name:
        db_name = os.getenv("DB_NAME", "").strip()

    if not db_name:
        db_name = "carepriority_gov"

    return mongo_uri, db_name


MONGO_URI, DB_NAME = get_mongo_config()

# Global in-memory storage for non-streamlit testing or fallback
_FALLBACK_STORE: Dict[str, Dict[str, Any]] = {}
_MONGO_CLIENT = None
_MONGO_DB = None
_DB_MODE = None  # "mongodb" or "session"
_DB_STATUS_MSG = ""


def _init_in_memory_store():
    """Initializes in-memory storage in st.session_state if in Streamlit, else module dict."""
    global _FALLBACK_STORE
    if HAS_STREAMLIT and hasattr(st, "session_state"):
        if "carepriority_db" not in st.session_state:
            st.session_state["carepriority_db"] = {}
        return st.session_state["carepriority_db"]
    return _FALLBACK_STORE


def get_db_status() -> Dict[str, Any]:
    """
    Checks MongoDB connectivity with short timeout (1500ms).
    Returns mode and connection status banner text.
    If MongoDB is unreachable or MONGO_URI is unset, falls back seamlessly to session mode.
    """
    global _MONGO_CLIENT, _MONGO_DB, _DB_MODE, _DB_STATUS_MSG, MONGO_URI, DB_NAME

    active_uri, active_db_name = get_mongo_config()
    MONGO_URI = active_uri
    DB_NAME = active_db_name

    if _DB_MODE is not None:
        return {
            "mode": _DB_MODE,
            "connected": (_DB_MODE == "mongodb"),
            "banner": "Database: MongoDB connected" if _DB_MODE == "mongodb" else "Database: Local session mode (data resets on refresh)",
            "uri": active_uri,
            "db_name": active_db_name
        }

    if HAS_PYMONGO and active_uri:
        try:
            client = MongoClient(active_uri, serverSelectionTimeoutMS=1500)
            # Test connection
            client.admin.command('ping')
            _MONGO_CLIENT = client
            _MONGO_DB = client[active_db_name]
            _DB_MODE = "mongodb"
            _DB_STATUS_MSG = "Database: MongoDB connected"
            
            # Ensure indexes on status, tier, created_at
            _MONGO_DB.encounters.create_index([("status", ASCENDING)])
            _MONGO_DB.encounters.create_index([("tier", ASCENDING)])
            _MONGO_DB.encounters.create_index([("created_at", ASCENDING)])
            _MONGO_DB.encounters.create_index([("encounter_id", ASCENDING)], unique=True)
            
            return {
                "mode": "mongodb",
                "connected": True,
                "banner": _DB_STATUS_MSG,
                "uri": active_uri,
                "db_name": active_db_name
            }
        except Exception:
            _DB_MODE = "session"
            _DB_STATUS_MSG = "Database: Local session mode (data resets on refresh)"
    else:
        _DB_MODE = "session"
        _DB_STATUS_MSG = "Database: Local session mode (data resets on refresh)"

    return {
        "mode": "session",
        "connected": False,
        "banner": _DB_STATUS_MSG,
        "uri": active_uri,
        "db_name": active_db_name
    }


def insert_encounter(record: Dict[str, Any]) -> str:
    """Inserts a new encounter record into active database."""
    status_info = get_db_status()
    rec = copy.deepcopy(record)
    
    # Standardize created_at timestamp
    if "created_at" not in rec or not rec["created_at"]:
        rec["created_at"] = datetime.now().isoformat()
    elif isinstance(rec["created_at"], datetime):
        rec["created_at"] = rec["created_at"].isoformat()

    encounter_id = rec["encounter_id"]

    if status_info["mode"] == "mongodb" and _MONGO_DB is not None:
        try:
            _MONGO_DB.encounters.update_one(
                {"encounter_id": encounter_id},
                {"$set": rec},
                upsert=True
            )
            return encounter_id
        except Exception:
            pass  # Fallback to in-memory if transient error

    store = _init_in_memory_store()
    store[encounter_id] = rec
    return encounter_id


def get_encounter(encounter_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves an encounter by encounter_id."""
    status_info = get_db_status()
    if status_info["mode"] == "mongodb" and _MONGO_DB is not None:
        try:
            doc = _MONGO_DB.encounters.find_one({"encounter_id": encounter_id}, {"_id": 0})
            if doc:
                return doc
        except Exception:
            pass

    store = _init_in_memory_store()
    rec = store.get(encounter_id)
    return copy.deepcopy(rec) if rec else None


def update_encounter(encounter_id: str, updates: Dict[str, Any]) -> bool:
    """Updates fields of an existing encounter."""
    status_info = get_db_status()
    if status_info["mode"] == "mongodb" and _MONGO_DB is not None:
        try:
            res = _MONGO_DB.encounters.update_one(
                {"encounter_id": encounter_id},
                {"$set": updates}
            )
            return res.matched_count > 0
        except Exception:
            pass

    store = _init_in_memory_store()
    if encounter_id in store:
        store[encounter_id].update(updates)
        return True
    return False


def get_all_encounters(filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Returns all encounters matching optional filters."""
    status_info = get_db_status()
    if status_info["mode"] == "mongodb" and _MONGO_DB is not None:
        try:
            query = filters or {}
            cursor = _MONGO_DB.encounters.find(query, {"_id": 0}).sort("created_at", ASCENDING)
            return list(cursor)
        except Exception:
            pass

    store = _init_in_memory_store()
    encounters = list(store.values())
    if filters:
        filtered = []
        for enc in encounters:
            match = True
            for k, v in filters.items():
                if enc.get(k) != v:
                    match = False
                    break
            if match:
                filtered.append(copy.deepcopy(enc))
        return sorted(filtered, key=lambda x: x.get("created_at", ""))
    return [copy.deepcopy(x) for x in sorted(encounters, key=lambda x: x.get("created_at", ""))]


def get_queue(
    department_filter: Optional[str] = None,
    status_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Returns clinical queue ordered strictly according to requirements:
    (a) Needs Nurse Review / Abstained (always on top)
    (b) Priority Tier L1 -> L5 (L1 first, then L2, etc.)
    (c) Arrival time (created_at ascending)
    """
    all_records = get_all_encounters()

    # Apply filters
    filtered = []
    for r in all_records:
        r_status = r.get("status", "Waiting")
        if status_filter and status_filter != "All":
            if r_status != status_filter:
                continue
        elif not status_filter:
            # Default active queue: exclude Completed
            if r_status == "Completed":
                continue

        if department_filter and department_filter != "All":
            if r.get("department") != department_filter:
                continue

        filtered.append(r)

    # Queue Sorting Comparator
    def sort_key(item: Dict[str, Any]) -> Tuple[int, int, str]:
        # Priority group 1: Needs Nurse Review (abstained or status == 'Awaiting Nurse Review')
        is_review = (
            item.get("abstained", False) or 
            item.get("tier") is None or 
            item.get("status") == "Awaiting Nurse Review"
        )
        group_priority = 0 if is_review else 1

        # Priority group 2: Tier (1 to 5). If None, give 99
        tier_val = item.get("tier")
        tier_sort = tier_val if (tier_val is not None and isinstance(tier_val, int)) else 99

        # Priority group 3: Arrival time
        arrival = str(item.get("created_at", ""))

        return (group_priority, tier_sort, arrival)

    return sorted(filtered, key=sort_key)


def get_metrics(num_doctors: int = 2) -> Dict[str, Any]:
    """
    Computes real-time OPD dashboard metrics.
    Consistent across MongoDB and session fallback.
    """
    all_encounters = get_all_encounters()
    now = datetime.now()

    total_walkins = len(all_encounters)
    waiting_count = 0
    awaiting_nurse_review = 0
    in_consultation_count = 0
    completed_count = 0
    l1_count = 0
    l2_count = 0
    dept_counts: Dict[str, int] = {}
    wait_times_mins: List[float] = []

    retriage_due_count = 0

    for r in all_encounters:
        status = r.get("status", "Waiting")
        tier = r.get("tier")
        dept = r.get("department", "General Medicine")
        abstained = r.get("abstained", False)

        if status == "Completed":
            completed_count += 1
        elif status == "In Consultation":
            in_consultation_count += 1
        elif status == "Awaiting Nurse Review" or abstained or tier is None:
            awaiting_nurse_review += 1
            waiting_count += 1
        else:
            waiting_count += 1

        # Department queue load for currently active/waiting patients
        if status in ["Waiting", "Awaiting Nurse Review"]:
            dept_counts[dept] = dept_counts.get(dept, 0) + 1
            is_due, _, _ = check_retriage_due(r, now=now)
            if is_due:
                retriage_due_count += 1

        # Active Level 1 and 2 emergency badges
        if status in ["Waiting", "Awaiting Nurse Review", "In Consultation"]:
            if tier == 1:
                l1_count += 1
            elif tier == 2:
                l2_count += 1

        # Wait time calculation for uncompleted patients
        if status in ["Waiting", "Awaiting Nurse Review"]:
            created_str = r.get("created_at")
            if created_str:
                try:
                    dt = datetime.fromisoformat(created_str)
                    mins = max(0.0, (now - dt).total_seconds() / 60.0)
                    wait_times_mins.append(mins)
                except Exception:
                    pass

    avg_wait = (sum(wait_times_mins) / len(wait_times_mins)) if wait_times_mins else 0.0
    active_load = waiting_count + in_consultation_count
    safe_docs = max(1, num_doctors)
    patients_per_doctor = round(active_load / safe_docs, 1)

    return {
        "total_walkins": total_walkins,
        "waiting_count": waiting_count,
        "awaiting_nurse_review": awaiting_nurse_review,
        "in_consultation_count": in_consultation_count,
        "completed_count": completed_count,
        "average_wait_mins": round(avg_wait, 1),
        "l1_count": l1_count,
        "l2_count": l2_count,
        "retriage_due_count": retriage_due_count,
        "dept_load": dept_counts,
        "patients_per_doctor": patients_per_doctor,
        "num_doctors": safe_docs
    }


RETRIAGE_THRESHOLDS_MINS: Dict[Optional[int], int] = {
    1: 0,    # Immediate Resuscitation (0m wait)
    2: 15,   # Emergent: re-check vitals every 15m
    3: 60,   # Urgent: re-check vitals every 60m
    4: 120,  # Less Urgent: re-check vitals every 120m
    5: 180,  # Non-Urgent: re-check vitals every 180m
    None: 30 # Needs Nurse Review: re-check every 30m
}


def check_retriage_due(encounter: Dict[str, Any], now: Optional[datetime] = None) -> Tuple[bool, int, int]:
    """
    Checks if an encounter is due for clinical re-triage based on wait duration.
    Returns (is_due, wait_mins, threshold_mins).
    """
    status = encounter.get("status", "Waiting")
    if status not in ["Waiting", "Awaiting Nurse Review"]:
        return (False, 0, 999)

    tier = encounter.get("tier")
    threshold = RETRIAGE_THRESHOLDS_MINS.get(tier, 60)
    current_time = now or datetime.now()

    ref_time_str = encounter.get("last_retriage_at") or encounter.get("created_at")
    wait_mins = 0
    if ref_time_str:
        try:
            dt = datetime.fromisoformat(ref_time_str)
            wait_mins = max(0, int((current_time - dt).total_seconds() / 60))
        except Exception:
            pass

    is_due = (wait_mins >= threshold)
    return (is_due, wait_mins, threshold)


def re_triage_encounter(
    encounter_id: str,
    new_vitals: Dict[str, Any],
    complaint_update: Optional[str] = None,
    actor: str = "Triage Nurse (Re-assessment)"
) -> Optional[Dict[str, Any]]:
    """
    Executes a clinical re-triage assessment for a waiting patient.
    Re-runs the hybrid triage pipeline with updated vitals/symptoms,
    escalates if deteriorated, records the audit trail, and resets re-triage timer.
    """
    from triage_engine import run_triage

    enc = get_encounter(encounter_id)
    if not enc:
        return None

    old_tier = enc.get("tier")
    merged_vitals = copy.deepcopy(enc.get("vitals", {}))
    for k, v in new_vitals.items():
        if v is not None:
            merged_vitals[k] = v

    full_complaint = enc.get("complaint_original", "")
    if complaint_update and complaint_update.strip():
        full_complaint = f"{full_complaint} [RE-TRIAGE UPDATE: {complaint_update.strip()}]"

    triage_result = run_triage(
        patient_id=encounter_id,
        patient_name=enc.get("patient_name", ""),
        age=enc.get("age"),
        gender=enc.get("gender", "Unknown"),
        spo2=merged_vitals.get("spo2"),
        bp_systolic=merged_vitals.get("bp_systolic"),
        bp_diastolic=merged_vitals.get("bp_diastolic"),
        pulse=merged_vitals.get("pulse"),
        temperature=merged_vitals.get("temperature"),
        complaint=full_complaint,
        known_conditions=enc.get("known_conditions", ""),
        medications=enc.get("medications", ""),
        temp_unit="F"
    )

    new_tier = triage_result["tier"]
    now_iso = datetime.now().isoformat()

    deteriorated = False
    if old_tier is not None and new_tier is not None and new_tier < old_tier:
        deteriorated = True

    current_audit = enc.get("audit", [])
    current_audit.append({
        "timestamp": now_iso,
        "actor": actor,
        "action": "Clinical Re-Triage" + (" (Deterioration Escalated)" if deteriorated else ""),
        "old_tier": old_tier,
        "new_tier": new_tier,
        "reason": f"Clinical re-triage. {complaint_update or ''}".strip(),
        "updated_vitals": triage_result["cleaned_vitals"]
    })

    updates = {
        "vitals": triage_result["cleaned_vitals"],
        "complaint_original": full_complaint,
        "tier": new_tier,
        "tier_name": triage_result["tier_name"],
        "department": triage_result["department"],
        "abstained": triage_result["abstained"],
        "abstention_reason": triage_result["abstention_reason"],
        "red_flags": triage_result["red_flags"],
        "reasons": triage_result["reasons"],
        "is_escalated": triage_result["is_escalated"],
        "rule_triggered": triage_result["escalation_rule"],
        "last_retriage_at": now_iso,
        "audit": current_audit
    }

    update_encounter(encounter_id, updates)
    return get_encounter(encounter_id)


def bulk_insert(records: List[Dict[str, Any]]) -> int:
    """Inserts a batch of encounters."""
    status_info = get_db_status()
    now_iso = datetime.now().isoformat()
    clean_records = []
    
    for r in records:
        c = copy.deepcopy(r)
        if "created_at" not in c or not c["created_at"]:
            c["created_at"] = now_iso
        elif isinstance(c["created_at"], datetime):
            c["created_at"] = c["created_at"].isoformat()
        clean_records.append(c)

    if status_info["mode"] == "mongodb" and _MONGO_DB is not None:
        try:
            # Upsert each to avoid duplicate key errors
            for c in clean_records:
                _MONGO_DB.encounters.update_one(
                    {"encounter_id": c["encounter_id"]},
                    {"$set": c},
                    upsert=True
                )
            return len(clean_records)
        except Exception:
            pass

    store = _init_in_memory_store()
    for c in clean_records:
        store[c["encounter_id"]] = c
    return len(clean_records)


def clear_all() -> bool:
    """Clears all records for demo reset or stress test."""
    status_info = get_db_status()
    if status_info["mode"] == "mongodb" and _MONGO_DB is not None:
        try:
            _MONGO_DB.encounters.delete_many({})
        except Exception:
            pass

    store = _init_in_memory_store()
    store.clear()
    global _FALLBACK_STORE
    _FALLBACK_STORE.clear()
    return True
