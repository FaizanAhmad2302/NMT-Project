import pandas as pd
import json
import os

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "cleaned_final.csv")
SPLITS_PATH = os.path.join(DATA_DIR, "splits.json")

# Load dataset with UTF-8 encoding
df = pd.read_csv(CSV_PATH, encoding="utf-8")[["Urdu", "RomanUrdu"]]

# Load splits.json
with open(SPLITS_PATH, "r", encoding="utf-8") as f:
    splits = json.load(f)

# Export separate CSVs with UTF-8 encoding
df.iloc[splits["train"]].to_csv(os.path.join(DATA_DIR, "train.csv"), index=False, encoding="utf-8-sig")
df.iloc[splits["val"]].to_csv(os.path.join(DATA_DIR, "val.csv"), index=False, encoding="utf-8-sig")
df.iloc[splits["test"]].to_csv(os.path.join(DATA_DIR, "test.csv"), index=False, encoding="utf-8-sig")

print(f"✅ Saved train.csv ({len(splits['train'])})")
print(f"✅ Saved val.csv   ({len(splits['val'])})")
print(f"✅ Saved test.csv  ({len(splits['test'])})")
