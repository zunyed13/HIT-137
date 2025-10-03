from functools import wraps
import time


def timed(fn):
    """Decorator to measure runtime and attach seconds to return (if dict-like)."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        out = fn(*args, **kwargs)
        elapsed = time.time() - start
        try:
            # best-effort logging on adapters that expose .log
            self = args[0]
            if hasattr(self, "log"):
                self.log(f"{fn.__name__}() finished in {elapsed:.2f}s")
        except Exception:
            pass
        return out
    return wrapper


def requires_input(fn):
    """Decorator to guard against empty/None inputs to .run()."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # heuristics: first arg after self is the input we care about
        if len(args) < 2 and not kwargs:
            raise ValueError("Input required")
        val = args[1] if len(args) >= 2 else next(iter(kwargs.values()))
        if val is None:
            raise ValueError("Input cannot be None")
        if isinstance(val, str) and not val.strip():
            raise ValueError("Input string cannot be empty")
        return fn(*args, **kwargs)
    return wrapper
