import json
from pathlib import Path
from src.syllogism_logic import SyllogismLogic

BASE = Path(__file__).resolve().parent.parent
POLISHED = BASE / "data" / "polished" / "polished_syllogisms_variables.json"
TRAIN = BASE / "train_data.json"
OUT = BASE / "predictions" / "derived_validity.json"

def main():
    with open(POLISHED, "r", encoding="utf-8") as f:
        polished = json.load(f)
    with open(TRAIN, "r", encoding="utf-8") as f:
        gt = {ex["id"]: ex["validity"] for ex in json.load(f)}

    total, correct = 0, 0
    results = []

    for ex in polished:
        sid = ex["id"]
        parts = [p.strip() for p in ex["syllogism"].split(".") if p.strip()]
        if len(parts) != 3:
            continue
        major, minor, concl = [p + "." for p in parts]
        try:
            pred = SyllogismLogic.is_valid(major, minor, concl)
        except Exception as e:
            pred = None
        true = gt.get(sid)
        match = (pred == true)
        total += 1
        correct += match
        results.append({"id": sid, "pred": pred, "true": true, "match": match})

    acc = correct / total if total else 0.0
    print(f"Formal validity accuracy: {acc:.3f} ({correct}/{total})")

    OUT.parent.mkdir(exist_ok=True, parents=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "results": results}, f, indent=2)

if __name__ == "__main__":
    main()
