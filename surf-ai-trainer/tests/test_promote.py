from app.promote import maybe_promote


def test_rejects_when_new_macro_f1_is_much_worse():
    new_metrics = {"macro_f1": 0.60}
    active_metrics = {"macro_f1": 0.70}
    assert maybe_promote(new_metrics, active_metrics) is False


def test_accepts_when_new_macro_f1_is_equal_or_better():
    active_metrics = {"macro_f1": 0.70}
    assert maybe_promote({"macro_f1": 0.70}, active_metrics) is True
    assert maybe_promote({"macro_f1": 0.75}, active_metrics) is True


def test_accepts_when_no_active_version_yet():
    assert maybe_promote({"macro_f1": 0.01}, None) is True
