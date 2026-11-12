"""Generator / injector / feature-builder behaviour, incl. reproducibility."""
import hashlib
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
