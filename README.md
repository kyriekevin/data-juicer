# LLM Data Recipe

LLM Data Recipe is a collection of data processing scripts for large language models. It provides a set of data processing methods for large language models, including data synthesis, data selection, and data conversion.

## Datasets

### Download

There are two ways to download datasets:

#### HF-Mirror

We use [HF-Mirror](https://hf-mirror.com/) to download original datasets. And we provide scripts to download datasets to local storage.

First, we need to install the dependencies:

```bash
pip install -U huggingface_hub

# or
poetry add huggingface_hub
```

Then we need to set the environment variable `HF_ENDPOINT`, you'd better set it in your `.bashrc` or `.zshrc`.

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

Finally, we can download the datasets:
```bash
huggingface-cli download --repo-type dataset --resume-download wikitext --local-dir wikitext
```

#### ModelScope

We use ModelScope to download original datasets. And we provide scripts to download datasets to local storage.

First, we need to install the dependencies:

```bash
pip install -U modelscope

# or
poetry add modelscope
```

Then we can download the datasets:
```
modelscope download --dataset DATASET_NAME --local_dir LOCAL_DIR
```

Because Modelscope is a domestic source, the download speed is faster than HF-Mirror. The speed difference in model downloading is quite obvious.

### Data Storage Structure

We use the following structure to store the datasets:
For original datasets, we store them in `raw` directory.
For intermediate datasets(after preprocess), we store them in `interim` directory.
For processed datasets(after sampling), we store them in `processed` directory.

```
data
├── raw
|   └── domain
|       └── dataset_name
|           └── xxx.xxx
├── interim
│   └── version
│       └── domain
│           └── dataset_name
│               └── part-xxx.jsonl
└── processed
    └── version
        └── domain.jsonl
```

### Preprocess

Data preprocessing mainly involves single-round and multi-round data conversion, as well as metadata information maintenance.

### Process

#### Data synthesis

Two methods are illustrated for data synthesis: Magpie and Personas.
* Magpie uses the model's continuation capability to generate prompts, and then generates responses based on the prompts. The prompt distribution may be closer to the prompt distribution during annealing.
* The Personas method takes into account the diversity brought by persona information, and answers questions from different perspectives for different personas, thereby improving diversity.

#### Data selection

The ifd method is used for data selection. The ifd method mainly considers the difficulty of the model answering questions to choose whether to train.

#### Data conversion

Data conversion mainly involves converting high-quality SFT data into pretrain format for annealing training to improve model performance.
