"""No-op ipywidgets stub (headless generation doesn't need notebook widgets).
Any attribute access / call returns another no-op, so `widgets.X(...)` is safe.
"""
class _NoOp:
    def __getattr__(self, name): return _NoOp()
    def __call__(self, *a, **k): return _NoOp()
    def __iter__(self): return iter(())
    def __enter__(self): return self
    def __exit__(self, *a): return False

widgets = _NoOp()

def __getattr__(name):   # module-level: ipywidgets.AnyOtherName
    return _NoOp()
