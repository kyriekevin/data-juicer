from typing import Optional

import torch


def get_perplexity(
    tokenizer,
    model,
    text: str,
    max_length: int,
    device: str,
    target_span: Optional[str] = None,
) -> tuple[float, float]:
    """
    Get the perplexity of the given text.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to encode the text.
        model (transformers.PreTrainedModel): Model to compute the perplexity.
        text (str): Text to compute the perplexity.
        max_length (int): Maximum length of the input sequence.
        device (str): Device to be used for training and inference.
        target_span (Optional[str], optional): Target span to mask. Defaults to None.

    Returns:
        tuple[float, float]: Perplexity and loss of the given text.
    """

    try:
        model = model.to(device)
        input_ids = tokenizer.encode(
            text, return_tensors="pt", truncation=True, max_length=max_length
        ).to(device)

        if target_span is not None:
            start_index = text.rfind(target_span)
            start_token = len(tokenizer.encode(text[:start_index]))

            labels = input_ids.clone()
            labels[0, :start_token] = -100
        else:
            labels = input_ids.contiguous()

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to("cpu").item(), loss.to("cpu").item()

    except:
        return 0, 0
