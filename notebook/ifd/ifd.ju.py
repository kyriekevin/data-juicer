# %%
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.ifd import ifd_score
from src.utils.file_io import read_file
from src.utils.util import get_device

# %%
MODEL_PATH = os.environ["MODEL_DIR"]
DATA_PATH = os.environ["DATA_DIR"]

model_path = MODEL_PATH + "/gpt2"
data_path = DATA_PATH + "/raw/General/alpaca_data.json"

# %%
device = get_device()
model = AutoModelForCausalLM.from_pretrained(model_path).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
max_length = 1024
data = read_file(data_path, 0, 1000)

# %%
ifd_data = ifd_score(model, tokenizer, max_length, device, data, "alpaca")

df = pd.DataFrame([{"ifd": item["metadata"]["ifd"], "data": item} for item in ifd_data])
df = df.dropna(subset=["ifd"])

stats = df["ifd"].describe()
print(stats)

# %%
correct = len(df[(df["ifd"] > 0) & (df["ifd"] < 1)]) / len(df)
print(f"Error rate: {1 - correct:.2f}")

print(df[df["ifd"] > 1].iloc[0]["data"])

# %%
df = df[(df["ifd"] < 1) & (df["ifd"] > 0)]
plt.figure(figsize=(10, 6))
sns.violinplot(y=df["ifd"])
plt.xlabel("alpaca")
plt.ylabel("IFD")
plt.tight_layout()
plt.show()

# %%
sample_rate = 0.2
top_20_percent = df.nlargest(n=int(len(df) * sample_rate), columns="ifd")
print(top_20_percent["ifd"].describe())

# %%
print(top_20_percent["data"].iloc[0])
