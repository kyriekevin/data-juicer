import argparse

from src.utils.file_io import read_file, write_file
from src.utils.format_convert import sft2pt


def args_parse():
    """
    Parse the command line arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        type=str,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = args_parse()
    sft_data = read_file(args.input_path)
    pt_data = sft2pt(sft_data)
    write_file(args.output_path, pt_data)
