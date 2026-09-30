"""
Splits the 117-question out-of-domain set into two halves, the same way
train/val/test were split, and for the same reason: a threshold must never
be chosen by looking at the data it will be reported against.

- out_of_domain_calibration.json: used, together with val, to pick a better
  threshold for the hybrid gate now that we know its current one (0.34) was
  tuned on too small and too easy a set.
- out_of_domain_heldout.json: never touched during calibration. The final
  refusal rate is reported only against this file.

Run with: python src/split_out_of_domain_set.py
"""

import json
import random
from pathlib import Path

SOURCE_PATH = Path("data/out_of_domain_test.json")
CALIBRATION_PATH = Path("data/out_of_domain_calibration.json")
HELDOUT_PATH = Path("data/out_of_domain_heldout.json")
SEED = 42


def main():
    with open(SOURCE_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    rng = random.Random(SEED)
    shuffled = questions[:]
    rng.shuffle(shuffled)

    half = len(shuffled) // 2
    calibration = shuffled[:half]
    heldout = shuffled[half:]

    with open(CALIBRATION_PATH, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    with open(HELDOUT_PATH, "w", encoding="utf-8") as f:
        json.dump(heldout, f, indent=2)

    print(f"Total: {len(questions)} -> calibration: {len(calibration)}, held-out: {len(heldout)}")
    print(f"Saved to {CALIBRATION_PATH} and {HELDOUT_PATH}")


if __name__ == "__main__":
    main()
