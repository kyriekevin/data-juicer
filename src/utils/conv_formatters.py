from typing import Any, Dict, List


def format_single_turn_conv(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format data for single turn conversation

    Args:
        data (Dict[str, Any]): data to format

    Returns:
        Dict[str, Any]: data formatted
    """

    conv = []
    system_prompt = data.get("system", "")
    input_prompt = data.get("input", "")
    output_prompt = data.get("output", "")

    if input_prompt == "" or output_prompt == "":
        raise ValueError("Input or output prompt is empty")

    conv.append({"from": "human", "value": input_prompt})
    conv.append({"from": "gpt", "value": output_prompt})

    if system_prompt:
        return {"conversations": conv, "system": system_prompt}

    return {"conversations": conv}


def format_multi_turn_conv(
    data: List[Dict[str, Any]], role_key: str = None, value_key: str = None
) -> Dict[str, Any]:
    """
    Format data for multi turn conversation

    Args:
        data (List[Dict[str, Any]]): data to format
        role_key (str): key to extract role from data
        value_key (str): key to extract value from data

    Returns:
        Dict[str, Any]: data formatted
    """

    conv = []
    last_role = ""
    system_prompt = ""
    role_key = role_key or "role"
    value_key = value_key or "content"

    role_mapping = {"user": "human", "assistant": "gpt"}

    for item in data:
        role = item.get(role_key, "")
        role = role_mapping.get(role, role)
        value = item.get(value_key, "")

        if last_role == role:
            raise ValueError(f"Role {role} is repeated")

        if role == "system":
            system_prompt = value
        elif role in role_mapping.values():
            if value == "":
                raise ValueError(f"Value for role {role} is empty")
            conv.append({"from": role, "value": value})

        last_role = role

    if last_role == "human":
        raise ValueError("Human role is the last role")

    if system_prompt:
        return {"conversations": conv, "system": system_prompt}

    return {"conversations": conv}
