from contextlib import contextmanager

_grad_enabled = True
# create npg.no_grad context manager

@contextmanager
def no_grad():
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev

def is_grad_enabled():
    return _grad_enabled