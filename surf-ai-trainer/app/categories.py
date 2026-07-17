# Wire contract - must match dev.slne.surf.ai.api.model.AiCategory ordinal order exactly.
CATEGORIES = ["HARASSMENT", "SELF_HARM", "HATE_SPEECH", "SEXUAL", "THREAT", "CHILD_SAFETY"]


def labels_to_vector(labels: list[str]) -> list[float]:
    return [1.0 if c in labels else 0.0 for c in CATEGORIES]
