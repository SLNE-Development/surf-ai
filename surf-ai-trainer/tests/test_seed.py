from app.seed import load_seed

VALID_CATEGORIES = {
    "HARASSMENT",
    "SELF_HARM",
    "HATE_SPEECH",
    "SEXUAL",
    "THREAT",
    "CHILD_SAFETY",
}

REQUIRED_EXAMPLES = {
    "kys": {"SELF_HARM"},
    "geh dich doch einfach umbringen": {"SELF_HARM"},
    "du hurensohn": {"HARASSMENT"},
    "penis": {"SEXUAL"},
}


def test_seed_corpus_has_enough_rows():
    rows = load_seed()
    assert len(rows) > 200


def test_all_labels_are_valid_categories():
    rows = load_seed()
    for row in rows:
        for label in row["labels"]:
            assert label in VALID_CATEGORIES


def test_required_examples_present():
    rows = load_seed()
    by_text = {row["text"]: set(row["labels"]) for row in rows}
    for text, expected_labels in REQUIRED_EXAMPLES.items():
        assert text in by_text, f"missing required example: {text!r}"
        assert by_text[text] == expected_labels


def test_has_minecraft_hard_negatives():
    rows = load_seed()
    hard_negatives = [row for row in rows if row["source"] == "minecraft_hard_negative" and row["labels"] == []]
    assert len(hard_negatives) >= 10
