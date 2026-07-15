"""
A minimal stand-in for the `streamlit` module, covering only the exact
API surface app.py uses (set_page_config, cache_data/cache_resource
decorators, layout containers, widgets, dataframe/image/pyplot
display). Used ONLY by tests/smoke_test_dashboard.py to execute app.py
end-to-end and catch real runtime errors (bad file paths, wrong
function signatures, KeyErrors, etc.) in a sandbox where the real
`streamlit` package could not be installed (see docs/day7_progress.md -
its `pyarrow` dependency was too large to download here).

This does NOT render anything and does NOT verify visual output - it
only verifies the script runs top-to-bottom without exceptions. Real
UI verification requires `streamlit run app.py` on a machine with
normal internet access to install streamlit itself.
"""

from contextlib import contextmanager


def _noop(*args, **kwargs):
    return None


class _DummyContainer:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def container(self):
        return _DummyContainer()

    def empty(self):
        return _DummyContainer()

    def pyplot(self, fig=None, **kwargs):
        assert fig is not None
        return None

    def dataframe(self, df=None, **kwargs):
        assert df is not None
        return None

    def error(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        # delegate widget calls (slider, selectbox, button, metric, ...) to
        # the module-level stub of the same name, so `col.slider(...)`
        # behaves the same as `st.slider(...)` rather than silently
        # returning None regardless of arguments.
        global_fn = globals().get(name)
        if callable(global_fn):
            return global_fn
        return _noop


def cache_data(func=None, **kwargs):
    if func is None:
        return lambda f: f
    return func


def cache_resource(func=None, **kwargs):
    if func is None:
        return lambda f: f
    return func


def set_page_config(*args, **kwargs):
    return None


def selectbox(label, options, index=0, **kwargs):
    options = list(options)
    return options[index] if options else None


def slider(label, min_value=0, max_value=100, value=None, **kwargs):
    return value if value is not None else min_value


def columns(n, **kwargs):
    return tuple(_DummyContainer() for _ in range(n))


def tabs(names):
    return tuple(_DummyContainer() for _ in names)


@contextmanager
def spinner(text=""):
    yield


sidebar = _DummyContainer()


def dataframe(df, **kwargs):
    assert df is not None
    return None


def empty(**kwargs):
    return _DummyContainer()


def button(label, **kwargs):
    return True  # always "clicked" so the smoke test exercises the live-feed loop


success = error = _noop


def image(path, **kwargs):
    import os
    assert os.path.exists(path), f"image path missing: {path}"
    return None


def pyplot(fig, **kwargs):
    assert fig is not None
    return None


def metric(label, value, **kwargs):
    return None


title = header = subheader = caption = write = markdown = divider = _noop
