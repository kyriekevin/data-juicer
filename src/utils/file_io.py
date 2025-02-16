import glob
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from src.utils.util import read_json, read_jsonl

FILE_READERS: Dict[str, Callable[[Path], Any]] = {
    "json": read_json,
    "jsonl": read_jsonl,
    "parquet": pd.read_parquet,
}


def _read_file(
    file_path: Path, file_type: str
) -> Union[List[Dict[str, Any]], pd.DataFrame]:
    """
    Read a file based on its type.

    Args:
        file_path (Path): Path to the file.
        file_type (str): File type to read.

    Returns:
        Union[List[Dict[str, Any]], pd.DataFrame]: Data loaded from the file.
    """

    if file_type != file_path.suffix[1:]:
        raise ValueError(
            f"File type {file_path.suffix[1:]} does not match expected file type {file_type}."
        )

    reader = FILE_READERS.get(file_type)
    if reader is None:
        raise ValueError(f"Unsupported file type {file_type}.")

    return reader(file_path)


def read_file(
    file_path: Union[str, Path], start_idx: int = 0, end_idx: int = -1
) -> List[Dict[str, Any]]:
    """
    Load a JSON or a JSONL file which contains "sharegpt" or "alpaca" format data.

    Args:
        file_path (Union[str, Path]): The path to the JSON file.
        start_idx (int): The start index of the data to load. Defaults to 0.
        end_idx (int): The end index of the data to load. Defaults to -1.

    Returns:
        List[Dict[str, Any]]: The data loaded from the JSON file.
    """

    file_path = Path(file_path)
    file_type = file_path.suffix[1:].lower()

    if file_type not in ("json", "jsonl"):
        raise ValueError(f"Unsupported file format: {file_path}")

    data = _read_file(file_path, file_type)
    assert isinstance(data, list), f"Expected list but got {type(data)}"
    return data[start_idx:end_idx]


def read_files(
    input_folder: Union[str, Path], file_type: str
) -> Iterator[Tuple[str, Any]]:
    """
    Read all files of a specific type in a folder.

    Args:
        input_folder (Union[str, Path]): Folder to read files from.
        file_type (str): File type to read.

    Returns:
        Iterator of tuples containing the file name and the file content.
    """

    input_path = Path(input_folder).resolve()

    if not input_path.is_dir():
        if input_path.is_file() and input_path.suffix[1:] == file_type:
            yield input_path.name, _read_file(input_path, file_type)
        else:
            raise ValueError(f"{input_folder} is not a valid folder or file.")

    pattern = str(input_path / f"*{file_type}")

    for input_file in glob.iglob(pattern):
        file_path = Path(input_file).resolve()
        file_name = file_path.name
        content = _read_file(file_path, file_type)
        yield file_name, content


def write_file(output_path: Union[str, Path], data: List[Dict[str, Any]]) -> None:
    """
    Write data to a JSONL file.

    Args:
        output_path (Union[str, Path]): Path to write the file to.
        data (List[Dict[str, Any]]): Data to write.

    Returns:
        None
    """

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(data):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_files(
    output_folder: Union[str, Path],
    version: str,
    data: Dict[str, Dict[str, List[Dict[str, Any]]]],
    items_per_file: int = 100000,
) -> None:
    """
    Write data to multiple files.

    Args:
        output_folder (Union[str, Path]): Folder to write files to.
        version (str): Version of the data.
        data (Dict[str, Dict[str, List[Dict[str, Any]]]]): Data to save and wirte.
        items_per_file (int): Number of items to write per file.

    Returns:
        None
    """

    output_path = Path(output_folder).resolve()

    if not output_path.is_dir():
        raise ValueError(f"{output_folder} is not a valid folder.")

    output_path = output_path / version

    total_items = sum(
        len(items) for datasets in data.values() for items in datasets.values()
    )

    with tqdm(total=total_items, desc=f"Writing data to {output_path}") as pbar:
        for domain, datasets in data.items():
            for dataset, items in datasets.items():
                dataset_path = output_path / domain / dataset
                dataset_path.mkdir(parents=True, exist_ok=True)

                for i, chunk in enumerate(chunks(items, items_per_file)):
                    file_path = dataset_path / f"part-{i:03d}.jsonl"
                    with open(file_path, "w", encoding="utf-8") as f:
                        for item in chunk:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                            pbar.update(1)


def chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
    """
    Yield successive n-sized chunks from lst.

    Args:
        lst (List[Any]): List to chunk.
        n (int): Chunk size.

    Returns:
        Iterator of chunks.
    """

    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_custom_dataset(
    base_dir: str | Path, dataset_name: str, config_name: Optional[str] = None, **kwargs
):
    base_path = Path(base_dir)
    dataset_path = base_path / dataset_name

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dataset_path}")

    return load_dataset(str(dataset_path), config_name, **kwargs)
