from app.categories import CATEGORIES
from app.bootstrap_mapping import (
    map_jigsaw_toxic_row,
    map_jigsaw_unintended_bias_row,
    map_suicidewatch_row,
    map_germeval_row,
)


def test_jigsaw_toxic_maps_threat_and_insult():
    row = {
        "comment_text": "i will kill you",
        "toxic": "1",
        "severe_toxic": "0",
        "obscene": "0",
        "threat": "1",
        "insult": "0",
        "identity_hate": "0",
    }
    assert map_jigsaw_toxic_row(row) == ["THREAT"]


def test_jigsaw_toxic_maps_generic_toxicity_to_harassment():
    row = {
        "comment_text": "you are so stupid",
        "toxic": "1",
        "severe_toxic": "0",
        "obscene": "0",
        "threat": "0",
        "insult": "1",
        "identity_hate": "0",
    }
    assert map_jigsaw_toxic_row(row) == ["HARASSMENT"]


def test_jigsaw_toxic_clean_row_maps_to_no_categories():
    row = {
        "comment_text": "let's build a castle",
        "toxic": "0",
        "severe_toxic": "0",
        "obscene": "0",
        "threat": "0",
        "insult": "0",
        "identity_hate": "0",
    }
    assert map_jigsaw_toxic_row(row) == []


def test_jigsaw_unintended_bias_thresholds_continuous_scores():
    row = {
        "comment_text": "go kill yourself",
        "target": "0.8",
        "severe_toxicity": "0.1",
        "obscene": "0.0",
        "identity_attack": "0.0",
        "insult": "0.2",
        "threat": "0.9",
        "sexual_explicit": "0.0",
    }
    assert map_jigsaw_unintended_bias_row(row) == ["THREAT"]


def test_jigsaw_unintended_bias_below_threshold_maps_to_nothing():
    row = {
        "comment_text": "meh, whatever",
        "target": "0.3",
        "severe_toxicity": "0.1",
        "obscene": "0.0",
        "identity_attack": "0.0",
        "insult": "0.2",
        "threat": "0.1",
        "sexual_explicit": "0.0",
    }
    assert map_jigsaw_unintended_bias_row(row) == []


def test_suicidewatch_positive_class_maps_to_self_harm():
    assert map_suicidewatch_row({"text": "i want to end it all", "class": "suicide"}) == ["SELF_HARM"]


def test_suicidewatch_negative_class_maps_to_nothing():
    assert map_suicidewatch_row({"text": "just had a great day", "class": "non-suicide"}) == []


def test_germeval_offense_maps_to_harassment():
    row = {"text": "du bist so dumm", "task1_label": "OFFENSE", "task2_label": "INSULT"}
    assert map_germeval_row(row) == ["HARASSMENT"]


def test_germeval_other_maps_to_nothing():
    row = {"text": "schönes wetter heute", "task1_label": "OTHER", "task2_label": "OTHER"}
    assert map_germeval_row(row) == []


def test_all_mapped_categories_are_valid():
    valid = set(CATEGORIES)
    for labels in [
        map_jigsaw_toxic_row({"comment_text": "x", "toxic": "1", "severe_toxic": "1", "obscene": "1", "threat": "1", "insult": "1", "identity_hate": "1"}),
        map_germeval_row({"text": "x", "task1_label": "OFFENSE", "task2_label": "PROFANITY"}),
    ]:
        for label in labels:
            assert label in valid
