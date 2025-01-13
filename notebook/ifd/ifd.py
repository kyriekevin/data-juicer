import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.ifd import ifd_score
from src.utils.file_io import read_file, write_file
from src.utils.util import filter_convs_by_turns, get_device


def args_parse():
    """
    Parse the command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
    )
    parser.add_argument(
        "--input_path",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        type=str,
    )
    parser.add_argument("--data_format", type=str, default="sharegpt")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--max_length", type=int, default=1024)

    return parser.parse_args()


def main():
    args = args_parse()
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(args.model_path).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    data = read_file(args.input_path, args.start_idx, args.end_idx)
    data = filter_convs_by_turns(data, 1, 1)

    res = ifd_score(model, tokenizer, args.max_length, device, data, args.data_format)
    write_file(args.output_path, res)


if __name__ == "__main__":
    main()
