"""Generator / injector / feature-builder behaviour, incl. reproducibility."""
import hashlib
import pandas as pd
import tempfile

import generate_logs as gl
import inject_anomalies as ia
import feature_engineering as fe


def _build(seed, n_users=40, n_mal=4):
    d = tempfile.mkdtemp()
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        gl.main(out_dir=d, seed=seed, num_users=n_users)
        ia.main(data_dir=d, seed=seed, n_malicious=n_mal)
        feats, preview = fe.main(data_dir=d, write=False)
    return d, feats, preview


def test_pipeline_is_reproducible_for_a_seed():
    _, f1, _ = _build(5)
    _, f2, _ = _build(5)
    assert hashlib.md5(f1.to_csv().encode()).hexdigest() == hashlib.md5(f2.to_csv().encode()).hexdigest()


def test_different_seeds_give_different_data():
    _, f1, _ = _build(5)
    _, f2, _ = _build(6)
    assert not f1.equals(f2)


def test_injection_creates_expected_number_of_insiders():
    _, feats, preview = _build(9, n_users=40, n_mal=4)
    assert len(feats) == 40
    assert preview.is_malicious.fillna(False).astype(bool).sum() == 4


def test_after_hours_injection_changes_after_hours_feature():
    _, _, preview = _build(11, n_users=60, n_mal=5)
    ah = preview[preview.scenario == "after_hours_login"]
    normal = preview[~preview.is_malicious.fillna(False).astype(bool)]
    assert len(ah) >= 1
    assert ah.after_hours_logon_count.min() > normal.after_hours_logon_count.median()


def test_upload_analysis_roundtrip_matches_builtin_features():
    import glob, os
    import numpy as np
    import upload_analysis as ua
    d = "data/processed/full"
    files = {n: open(f"{d}/{n}", "rb").read() for n in ["logon.csv", "device.csv", "http.csv", "email.csv", "file.csv", "users.csv"]}
    feats, msgs = ua.prepare(files)
    ref = pd.read_csv(f"{d}/user_features.csv").set_index("user_id")
    assert list(feats.columns[1:]) == ua.FEATURE_COLUMNS
    assert np.allclose(feats.drop(columns="user_id").values, ref.loc[feats.user_id].values, atol=1e-6)


def test_upload_analysis_rejects_bad_files():
    import pytest
    import upload_analysis as ua
    with pytest.raises(ValueError):
        ua.prepare({"logon.csv": b"a,b\n1,2\n"})
    with pytest.raises(ValueError):
        ua.prepare({})
    small = "user_id," + ",".join(ua.FEATURE_COLUMNS) + "\nU1," + ",".join("1" for _ in ua.FEATURE_COLUMNS) + "\n"
    with pytest.raises(ValueError):
        ua.prepare({"features.csv": small.encode()})
