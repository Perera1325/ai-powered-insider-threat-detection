"""Dataset integrity + the 'no label leakage' guarantee."""
import pandas as pd

LOG_FILES = ["users", "logon", "device", "http", "email", "file", "labels", "user_features"]


def test_all_dataset_files_exist_and_nonempty():
    for name in LOG_FILES:
        df = pd.read_csv(f"data/processed/full/{name}.csv")
        assert len(df) > 0, name


def test_users_and_labels_cover_same_users(data):
    assert set(data["users"].user_id) == set(data["labels"].user_id) == set(data["features"].user_id)
    assert data["users"].user_id.is_unique


def test_exactly_the_injected_insiders_are_labelled(data):
    mal = data["labels"][data["labels"].is_malicious]
    assert len(mal) == 6
    assert set(mal.scenario) == {"after_hours_login", "mass_file_access", "usb_exfiltration", "data_exfil_email", "privilege_probe"}


def test_feature_table_has_no_labels_or_nans(data):
    f = data["features"]
    forbidden = {"is_malicious", "scenario", "injected_day", "label"}
    assert not forbidden & set(f.columns), "ground-truth leaked into model inputs"
    assert not f.isna().any().any()
    assert f.shape[1] - 1 == 28


def test_no_real_personal_data_markers(data):
    # all simulated mail stays on the fake internal domain or well-known free-mail domains
    allowed = ("simcorp-internal.local", "gmail.com", "yahoo.com", "outlook.com")
    for to in data["email_df"]["to"].dropna().head(2000):
        for rcpt in str(to).split(";"):
            assert rcpt.endswith(allowed), rcpt
