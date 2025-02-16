# %%
import json
import os
from pathlib import Path

from tqdm import tqdm

from src.utils.file_io import load_custom_dataset, write_file

DATA_DIR = Path(os.environ["DATA_DIR"]) / "raw" / "General"

# %%
try:
    ds = load_custom_dataset(DATA_DIR, "Personahub", "elite_persona")
    print(ds)
except Exception as e:
    print(f"Error loading dataset: {e}")
