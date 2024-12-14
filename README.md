# Data Juicer

Data Juicer is a project that collects and organizes open source data and refines it uniformly.

## Datasets

### Download

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
