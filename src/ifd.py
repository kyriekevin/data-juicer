import argparse
from typing import Any, Dict, List

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import PROMPT_DICT_NONE
from src.perplexity import get_perplexity
from src.utils.file_io import load_json
from src.utils.util import get_device


def args_parse():
    """
    Parse the command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=1024)

    return parser.parse_args()


def ifd_score(data: List[dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Calculate the IFD score of a list of data.

    Args:
        data (List[dict[str, str]]): A list of data, each data is a dict with keys "instruction", "input", and "output".

    Returns:
        List[Dict[str, Any]]: A list of data with keys "instruction", "input", "output", "ppl", and "loss".
    """

    res = []

    for i in tqdm(range(len(data))):
        data_i = data[i]
        instruct_i = data_i["instruction"]
        output_i = data_i["output"]
        input_i = data_i["input"] if "input" in data_i.keys() else ""

        if input_i == "":
            temp_dict = {"instruction": instruct_i}
            prompt_to_use = prompt_on_input.format_map(temp_dict)
        else:
            temp_dict = {"instruction": instruct_i, "input": input_i}
            prompt_to_use = prompt_input.format_map(temp_dict)

        whole_text = prompt_to_use + output_i
        instruct_i = prompt_to_use

        ppl_out_alone, loss_out_alone = get_perplexity(
            tokenizer, model, output_i, max_length, device
        )
        ppl_out_condition, loss_out_condition = get_perplexity(
            tokenizer, model, whole_text, max_length, device, output_i
        )

        temp_data_i = {
            "instruction": instruct_i,
            "input": input_i,
            "output": output_i,
            "ppl": [ppl_out_alone, ppl_out_condition],
            "loss": [loss_out_alone, loss_out_condition],
        }

        res.append(temp_data_i)

    return res


if __name__ == "__main__":
    args = args_parse()
    device = get_device()
    max_length = args.max_length
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, output_hidden_states=True
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    prompt_on_input = PROMPT_DICT_NONE["prompt_no_input"]
    prompt_input = PROMPT_DICT_NONE["prompt_input"]

    data = load_json(args.data_path, args.start_idx, args.end_idx)

    res = ifd_score(data)
    print(res)
