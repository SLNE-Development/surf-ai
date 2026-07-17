"""Maps raw rows from public toxicity/hate-speech corpora onto our 6 wire-contract
categories (app.categories.CATEGORIES). Each source has its own column schema and
labeling scheme; thresholds below are documented, auditable heuristics, not learned."""

THRESHOLD = 0.5


def _flag(value: str) -> bool:
    return float(value) >= THRESHOLD


def map_jigsaw_toxic_row(row: dict) -> list[str]:
    labels = []
    if _flag(row["threat"]):
        labels.append("THREAT")
    if _flag(row["obscene"]):
        labels.append("SEXUAL")
    if _flag(row["identity_hate"]):
        labels.append("HATE_SPEECH")
    if not labels and (_flag(row["insult"]) or _flag(row["toxic"]) or _flag(row["severe_toxic"])):
        labels.append("HARASSMENT")
    return labels


def map_jigsaw_unintended_bias_row(row: dict) -> list[str]:
    labels = []
    if _flag(row["threat"]):
        labels.append("THREAT")
    if _flag(row["sexual_explicit"]) or _flag(row["obscene"]):
        labels.append("SEXUAL")
    if _flag(row["identity_attack"]):
        labels.append("HATE_SPEECH")
    if not labels and (_flag(row["insult"]) or _flag(row["target"]) or _flag(row["severe_toxicity"])):
        labels.append("HARASSMENT")
    return labels


def map_suicidewatch_row(row: dict) -> list[str]:
    return ["SELF_HARM"] if row["class"].strip().lower() == "suicide" else []


def map_germeval_row(row: dict) -> list[str]:
    if row["task1_label"].strip().upper() != "OFFENSE":
        return []
    fine = row["task2_label"].strip().upper()
    if fine in ("INSULT", "ABUSE", "PROFANITY"):
        return ["HARASSMENT"]
    return ["HARASSMENT"]
