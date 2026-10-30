from types import SimpleNamespace

from pymongo.errors import PyMongoError

import db


class FailingCollection:
    def find_one(self, *_args, **_kwargs):
        raise PyMongoError("database unavailable")

    def find(self, *_args, **_kwargs):
        raise PyMongoError("database unavailable")

    def update_one(self, *_args, **_kwargs):
        raise PyMongoError("database unavailable")

    def delete_many(self, *_args, **_kwargs):
        raise PyMongoError("database unavailable")


def use_failing_mongo(monkeypatch):
    monkeypatch.setattr(db, "HAS_STREAMLIT", False)
    monkeypatch.setattr(db, "_DB_MODE", "mongodb")
    monkeypatch.setattr(db, "_DB_STATUS_MSG", "Database: MongoDB connected")
    monkeypatch.setattr(db, "_MONGO_CLIENT", None)
    monkeypatch.setattr(db, "_MONGO_DB", SimpleNamespace(encounters=FailingCollection()))
    monkeypatch.setattr(db, "_FALLBACK_STORE", {})


def test_insert_failure_switches_status_and_reads_from_session_store(monkeypatch):
    use_failing_mongo(monkeypatch)
    record = {"encounter_id": "ENC-FALLBACK-1", "tier": 2}

    assert db.insert_encounter(record) == "ENC-FALLBACK-1"
    status = db.get_db_status()
    saved_records = db.get_all_encounters()

    assert status["mode"] == "session"
    assert status["connected"] is False
    assert "MongoDB unavailable" in status["banner"]
    assert len(saved_records) == 1
    assert saved_records[0]["tier"] == 2


def test_bulk_insert_failure_switches_to_session_storage(monkeypatch):
    use_failing_mongo(monkeypatch)
    records = [{"encounter_id": "ENC-FALLBACK-2", "tier": 3}]

    assert db.bulk_insert(records) == 1
    assert db.get_db_status()["mode"] == "session"
    assert db.get_encounter("ENC-FALLBACK-2")["tier"] == 3


def test_read_and_update_failures_switch_to_session_storage(monkeypatch):
    use_failing_mongo(monkeypatch)
    db._FALLBACK_STORE["ENC-FALLBACK-4"] = {
        "encounter_id": "ENC-FALLBACK-4",
        "tier": 4,
    }

    assert db.get_encounter("ENC-FALLBACK-4")["tier"] == 4
    assert db.update_encounter("ENC-FALLBACK-4", {"tier": 2}) is True
    assert db.get_all_encounters()[0]["tier"] == 2
    assert db.get_db_status()["mode"] == "session"


def test_clear_reports_remote_failure_instead_of_success(monkeypatch):
    use_failing_mongo(monkeypatch)
    db._FALLBACK_STORE["ENC-FALLBACK-3"] = {"encounter_id": "ENC-FALLBACK-3"}

    assert db.clear_all() is False
    assert db.get_db_status()["mode"] == "session"
    assert db.get_all_encounters() == []
