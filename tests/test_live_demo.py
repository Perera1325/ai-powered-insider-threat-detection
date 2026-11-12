"""Live-demo simulation: risk of the chosen insider must jump after injection."""
import live_demo as ld
import dashboard_utils as du


def test_attack_injection_spikes_risk_score(data, models, matrix):
    X, cols, mean, std = matrix
    cohort = ld.create_live_cohort(n=12, seed=123)
    uid = cohort.user_id.iloc[2]
    states = list(ld.run_live_demo(cohort, cols, mean, std, models["Isolation Forest (sklearn)"],
                                   n_ticks=8, inject_at_tick=5, demo_user_id=uid, scenario="mass_file_access"))
    def risk(state):
        t = state["risk_table"]
        return float(t[t.user_id == uid].risk_score.iloc[0])
    before = max(risk(s) for s in states if s["tick"] < 5)
    after = risk(states[-1])
    assert states[4]["injected"] or states[5]["injected"]
    assert after > before, (before, after)
    assert after >= 70, after  # reaches HIGH risk
