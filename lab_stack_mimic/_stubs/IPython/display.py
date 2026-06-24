"""No-op IPython.display shims (headless: nothing to display)."""
def display(*args, **kwargs):
    pass
def clear_output(*args, **kwargs):
    pass
class HTML:
    def __init__(self, *a, **k): pass
class Markdown:
    def __init__(self, *a, **k): pass
