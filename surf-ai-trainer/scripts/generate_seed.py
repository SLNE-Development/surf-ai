"""
LLM-assisted / public-dataset seed generator (offline-runnable stub).

The committed seed corpus (surf-ai-trainer/seed/seed_{de,en,minecraft}.jsonl)
is hand-authored and intentionally small. To expand it with real-world
toxicity data, import one of the following public datasets - do NOT commit
their raw contents to this repo (license/size); instead run an import step
locally that maps rows into the same JSONL schema
({"text": str, "labels": [AiCategory...], "language": "de"|"en", "source": str})
and appends to the seed/*.jsonl files (or a new seed/seed_imported.jsonl).

Public datasets to consider:

1. Jigsaw Toxic Comment Classification (English)
   https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
   License: CC0 (via Kaggle competition rules - verify current terms before use).
   Columns: comment_text, toxic, severe_toxic, obscene, threat, insult,
   identity_hate. Suggested mapping:
     threat        -> THREAT
     identity_hate  -> HATE_SPEECH
     insult/toxic   -> HARASSMENT
     obscene        -> SEXUAL (only if sexual language, review manually)

2. GermEval 2018/2019 (German hate speech / offensive language)
   https://github.com/uds-lsv/GermEval-2018-Data
   License: research use - check the repo's terms before redistribution.
   Labels are coarse (OFFENSE/OTHER, sub-labels INSULT/ABUSE/PROFANITY);
   map INSULT -> HARASSMENT, ABUSE -> HATE_SPEECH as a first pass, then
   manually review sexual/threat/self-harm/child-safety examples since
   GermEval does not label those categories directly.

3. HASOC (Hate Speech and Offensive Content, multilingual incl. German)
   https://hasocfire.github.io/hasoc/
   License: research use, registration required per year's shared task.
   Provides HOF (Hate/Offensive) vs NOT labels plus a finer HATE/OFFN/PRFN
   split - map similarly to GermEval above.

None of the categories THREAT, SELF_HARM, SEXUAL, or CHILD_SAFETY are
reliably covered by the datasets above; those need either manual curation
or a separate specialized source, and human review either way before they
enter training data.

This script intentionally does not download or process anything by
default - it is documentation-as-code for the next import pass, run it
by hand once you've reviewed the license terms.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(
        "This is a documentation stub - see the module docstring for how to "
        "import Jigsaw/GermEval/HASOC data into the seed corpus. No data is "
        "downloaded automatically."
    )


if __name__ == "__main__":
    main()
