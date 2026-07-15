"""
Smoke test for app.py, using a stub `streamlit` module (see
_streamlit_stub.py) so the whole dashboard script's logic can be
executed and checked for runtime errors without needing the real
Streamlit package installed.

Run from the project root: python3 tests/smoke_test_dashboard.py
"""

import runpy
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.getcwd())

import _streamlit_stub  # noqa: E402
sys.modules["streamlit"] = _streamlit_stub

# selectbox needs to return a *specific* user id for the "explain an
# alert" section to exercise the SHAP/LIME code path realistically,
# rather than just whatever index=0 happens to be. Patch it to always
# pick the top-ranked alert, and the model dropdown to a fixed choice.
_original_selectbox = _streamlit_stub.selectbox


def _patched_selectbox(label, options, index=0, **kwargs):
    options = list(options)
    if "model" in label.lower():
        return "Isolation Forest"
    return options[0] if options else None


_streamlit_stub.selectbox = _patched_selectbox

print("Running app.py end-to-end with stubbed streamlit...")
runpy.run_path("app.py", run_name="__main__")
print("\nSMOKE TEST PASSED - app.py executed with no exceptions.")
