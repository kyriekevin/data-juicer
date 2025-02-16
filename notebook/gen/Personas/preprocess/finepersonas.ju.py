# %%
import json
import os
from pathlib import Path

from tqdm import tqdm

from src.utils.file_io import load_custom_dataset, write_file

DATA_DIR = Path(os.environ["DATA_DIR"]) / "raw" / "General"

# %%
try:
    ds = load_custom_dataset(DATA_DIR, "FinePersonas")
    print(ds)
except Exception as e:
    print(f"Error loading dataset: {e}")

# %%
res = []
for item in tqdm(ds["train"]):
    persona = item["persona"]
    topic = item["labels"]
    topic = json.loads(topic)
    if len(topic) == 0 or topic == "None":
        continue
    res.append({"persona": persona, "topic": topic})

# %%
write_file(str(DATA_DIR / "FinePersonas.jsonl"), res)
