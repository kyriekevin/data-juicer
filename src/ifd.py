from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from src.config import ALPACA_PROMPT, SHAREGPT_PROMPT


def get_perplexity(
    model,
    tokenizer,
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


def ifd_score(
    model,
    tokenizer,
    max_length: int,
    device: str,
    data: List[dict[str, Any]],
    format: str,
) -> List[Dict[str, Any]]:
    """
    Calculate the IFD score of a list of data.

    Args:
        model: The model to compute the IFD score.
        tokenizer: The tokenizer to encode the text.
        max_length (int): Maximum length of the sequence.
        device (str): Device to be used for training and inference.
        data (List[dict[str, Any]]): A list of data, each data is a dict which format depends on the format argument.
        format (str): The format of the data. It can be "alpaca" or "sharegpt". Defaults to "sharegpt".

    Returns:
        List[Dict[str, Any]]: A list of data with the IFD score.
    """

    res = []

    for i in tqdm(range(len(data))):
        data_i = data[i]

        if format == "alpaca":
            prompt_input = ALPACA_PROMPT["prompt_input"]
            prompt_no_input = ALPACA_PROMPT["prompt_no_input"]

            instruct_i = data_i["instruction"]
            output_i = data_i["output"]
            input_i = data_i["input"] if "input" in data_i.keys() else ""

            if input_i == "":
                temp_dict = {"instruction": instruct_i}
                prompt_to_use = prompt_no_input.format_map(temp_dict)
            else:
                temp_dict = {"instruction": instruct_i, "input": input_i}
                prompt_to_use = prompt_input.format_map(temp_dict)
        elif format == "sharegpt":
            prompt = SHAREGPT_PROMPT["prompt"]

            convs_i = data_i["conversations"]
            input_i = convs_i[0]["value"]
            output_i = convs_i[1]["value"]
            sys_i = data_i["system"]

            prompt_to_use = prompt.format_map({"system": sys_i, "input": input_i})
        else:
            raise ValueError("Invalid format.")

        whole_text = prompt_to_use + output_i
        instruct_i = prompt_to_use

        ppl_out_alone, loss_out_alone = get_perplexity(
            model, tokenizer, output_i, max_length, device
        )
        ppl_out_condition, loss_out_condition = get_perplexity(
            model, tokenizer, whole_text, max_length, device, output_i
        )

        temp_data_i = data_i.copy()
        temp_data_i["ppl"] = [ppl_out_alone, ppl_out_condition]
        temp_data_i["loss"] = [loss_out_alone, loss_out_condition]

        res.append(temp_data_i)

    return res
