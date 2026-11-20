"""Runs the REAL Streamlit app headlessly (streamlit.testing AppTest) - no stub."""
import os

import pytest
from streamlit.testing.v1 import AppTest

import dashboard_utils as du

T = 600
APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(APP, default_timeout=T)
    at.run()
    return at


def test_app_renders_without_exception(app):
    assert not app.exception, [e.value for e in app.exception]
    assert app.title[0].value == "AI-Powered Insider Threat Detection"


def test_alert_metrics_present(app):
    labels = [m.label for m in app.metric]
    for want in ("Users monitored", "Alerts shown", "High risk"):
        assert want in labels
    assert app.metric[0].value == "100"


def test_six_tabs(app):
    assert len(app.tabs) == 6


def test_explanation_figure_rendered_for_selected_alert(app):
    assert any("Explain an alert" in h.value for h in app.subheader)
    assert len(app.image) >= 1  # st.pyplot figures (SHAP/LIME + neighbourhood) are rendered


@pytest.mark.parametrize("model", du.MODEL_CHOICES)
def test_every_model_choice_renders(model):
    at = AppTest.from_file(APP, default_timeout=T)
    at.run()
    at.sidebar.selectbox[0].select(model).run()
    assert not at.exception, (model, [e.value for e in at.exception])


def test_live_demo_button_runs_and_raises_alert():
    at = AppTest.from_file(APP, default_timeout=T)
    at.run()
    # speed slider is the last slider in the Live Demo tab; make it fast
    speed = [s for s in at.slider if s.label.startswith("Seconds per simulated day")][0]
    speed.set_value(0.2)
    start = [b for b in at.button if "Start Live Feed" in b.label][0]
    start.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("ALERT" in e.value for e in at.error)
    assert any("Live feed complete" in e.value for e in at.success)


def test_realtime_tab_streams_scores_and_alerts_on_injected_attack():
    at = AppTest.from_file(APP, default_timeout=T)
    at.run()
    [b for b in at.button if "Start stream" in b.label][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    try:
        # let a few ticks flow; each .run() polls the scorer once (fragment auto-refresh is browser-side)
        import time
        deadline = time.time() + 120
        injected = False
        while time.time() < deadline:
            time.sleep(1.5)
            at.run()
            assert not at.exception, [e.value for e in at.exception]
            ticks = [m for m in at.metric if m.label == "Tick"]
            if ticks and int(ticks[0].value) >= 3 and not injected:
                [b for b in at.button if "Inject attack now" in b.label][0].click().run()
                injected = True
            if any("ALERT" in e.value for e in at.error):
                break
        alerts = [e.value for e in at.error if "ALERT" in e.value]
        assert alerts, "no alert raised after injecting an attack"
        assert any("matches the injected attack" in a for a in alerts)
    finally:
        [b for b in at.button if "Stop" in b.label][0].click().run()
