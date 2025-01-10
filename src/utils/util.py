import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def get_device():
    """
    Get the device to be used for training and inference.

    Args:
        None

    Returns:
        device (str): Device to be used for training and inference.
    """

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device


def read_json(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def filter_convs_by_turns(
    data: List[Dict[str, Any]], min_turns: int, max_turns: int
) -> List[Dict[str, Any]]:
    """
    Filter conversations by the number of turns.

    Args:
        data (List[Dict[str, Any]]): List of conversations.
        min_turns (int): Minimum number of turns.
        max_turns (int): Maximum number of turns.

    Returns:
        List[Dict[str, Any]]: List of conversations that have turns within the specified range.
    """

    filtered_convs = []

    for i in range(len(data)):
        convs = data[i]["conversations"]
        turns = len(convs) // 2
        if min_turns <= turns <= max_turns:
            filtered_convs.append(data[i])

    return filtered_convs
