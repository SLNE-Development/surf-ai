def maybe_promote(
    new_metrics: dict[str, float],
    active_metrics: dict[str, float] | None,
    min_macro_f1_delta: float = -0.02,
) -> bool:
    """Promote only if the new head's macro-F1 is not worse than the active head's by more
    than min_macro_f1_delta. With no active version yet (bootstrap), always promote."""
    if active_metrics is None:
        return True

    delta = new_metrics["macro_f1"] - active_metrics["macro_f1"]
    return delta >= min_macro_f1_delta
