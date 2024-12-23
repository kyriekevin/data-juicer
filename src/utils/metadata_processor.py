from enum import Enum
from typing import Any, Dict, Union


class Domain(Enum):
    MATH = "Math"
    CODE = "Code"
    FOLLOWING = "Following"
    REASONING = "Reasoning"
    GENERAL = "General"


class Language(Enum):
    EN = "En"
    ZH = "Zh"


def add_metadata(
    item: Dict[str, Any],
    domain: Union[Domain, str],
    dataset: str,
    source: str = None,
    lang: Union[Language, str] = None,
    quality: str = None,
) -> Dict[str, Any]:
    """
    Add metadata to a dictionary.

    Args:
        item: Dictionary to add metadata to.
        domain: Domain of the item.
        dataset: Dataset name.
        source: Source of the item. Defaults to None.
        lang: Language of the item. Defaults to None.
        quality: Quality of the item. Defaults to None.

    Returns:
        Dictionary with metadata added.
    """
    def process_enum(value, enum_class):
        if isinstance(value, enum_class):
            return value.value
        if isinstance(value, str):
            return enum_class[value.upper()].value if value.upper() in enum_class.__members__ else value
        return value

    metadata = {
        "domain": process_enum(domain, Domain),
        "dataset": dataset,
        **{k: v for k, v in [
            ("source", source),
            ("lang", process_enum(lang, Language)),
            ("quality", quality)
        ] if v is not None}
    }

    item["metadata"] = metadata
    return item
