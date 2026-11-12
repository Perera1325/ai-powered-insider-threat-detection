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


def test_four_tabs(app):
    assert len(app.tabs) == 4


def test_explanation_figure_rendered_for_selected_alert(app):
    assert any("Explain an alert" in h.value for h in app.subheader)
    assert len(app.image) >= 1  # st.pyplot figures (SHAP/LIME + neighbourhood) are rendered


@pytest.mark.parametrize("model", du.MODEL_CHOICES)
def test_every_model_choice_renders(model):
    at = AppTest.from_file(APP, default_timeout=T)
    at.run()
    at.sidebar.selectbox[0].select(model).run()
    assert not at.exception, (model, [e.value for e in at.exception])


def test_scratch_explanation_engine_renders():
    at = AppTest.from_file(APP, default_timeout=T)
    at.run()
    at.sidebar.radio[0].set_value("scratch").run()
    assert not at.exception


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
