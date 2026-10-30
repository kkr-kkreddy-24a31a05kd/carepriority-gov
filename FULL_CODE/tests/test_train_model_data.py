import pandas as pd

import train_model


def test_app_trainer_finds_nested_organizer_intake_and_labels(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    data_dir = project_dir / "data" / "organizer" / "data"
    data_dir.mkdir(parents=True)
    intake = pd.DataFrame({"visit_id": ["VISIT-1", "VISIT-2"], "chief_complaint": ["fever", "cough"]})
    labels = pd.DataFrame({"visit_id": ["VISIT-1", "VISIT-2"], "triage_level": [3, 4]})
    intake.to_csv(data_dir / "intake_train.csv", index=False)
    labels.to_csv(data_dir / "train_labels.csv", index=False)
    monkeypatch.setattr(train_model, "__file__", str(project_dir / "train_model.py"))

    merged, is_placeholder = train_model.find_data_file()

    assert is_placeholder is False
    assert merged["visit_id"].tolist() == ["VISIT-1", "VISIT-2"]
    assert merged["triage_level"].tolist() == [3, 4]
