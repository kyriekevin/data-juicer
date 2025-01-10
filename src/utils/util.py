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
