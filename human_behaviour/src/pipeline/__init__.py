__all__ = ["run_multicam_pipeline"]


def run_multicam_pipeline(*args, **kwargs):
    """Lazy wrapper — avoids loading heavy ML deps until actually called."""
    from .run_multicam_pipeline import run_multicam_pipeline as _run
    return _run(*args, **kwargs)
