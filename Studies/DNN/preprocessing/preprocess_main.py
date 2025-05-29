import argparse
import tomllib

from dataframe_parser import Parser
from dataset_builder import DatasetBuilder

### Util ###


def get_arguments():
    """
    Builds an argument parser to get CLI arguments
    """
    parser = argparse.ArgumentParser(
        prog="DataSetMaker",
    )
    parser.add_argument("-c", "--config", required=True, help="the .toml config file")
    parser.add_argument(
        "-i",
        "--input_files_dir",
        required=True,
        help="the directory containing the ROOT tuples to build datasets from",
    )
    args = parser.parse_args()
    return args


### Main ###

if __name__ == "__main__":

    # Read the CLI arguments
    args = get_arguments()

    # Read in config and datasets from args
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    parser = Parser(**config)
    print(f"Parsing file from {args.input_files_dir}...")
    df = parser.parse(args.input_files_dir)
    parser.save_result(df, config["output_directory"], config["run_name"] + "_df.pkl")

    builder = DatasetBuilder(df, **config)
    builder.build()
