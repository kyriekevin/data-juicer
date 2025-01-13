from typing import Any, Dict, List

from tqdm import tqdm


def sft2pt(sft_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert SFT data to PT data

    Args:
        sft_data (List[Dict[str, Any]]): SFT data

    Returns:
        List[Dict[str, Any]]: PT data
    """

    pt_data = []

    for data in tqdm(sft_data):

        content = ""

        if "system" in data and data["system"] != "You are a helpful assistant.":
            content += data["system"] + "\n" if data["system"] != "" else ""

        for conv in data["conversations"]:
            content += conv["value"] + "\n"

        if "metadata" in data:
            metadata = data["metadata"]
            pt_data.append({"metadata": metadata, "content": content})
        else:
            pt_data.append({"content": content})

    return pt_data
