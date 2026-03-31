import argparse
import os
import re
import shutil
from glob import glob

import uproot

### Constants
COMBINE_CMD = "combine -M Significance datacard.txt -t -1 --expectSignal=1"
COMBINE_OUTNAME = "higgsCombineTest.Significance.mH120.root"
BASE_DIR = "/afs/cern.ch/user/a/ayeagle/H_mumu/Studies/VBF_net/"
DATACARD = BASE_DIR + "datacard.txt"


def get_arguments():
    """
    Builds an argument parser to get CLI arguments for the config file and dataset directory.
    """
    parser = argparse.ArgumentParser(
        prog="Feature importance runner",
        description="Calculates Shapely values for the provided model and data",
    )
    parser.add_argument(
        "-r",
        "--results_dir",
        required=True,
        help="Path to the results directory containing evaluated_testing_df and model",
    )
    args = parser.parse_args()
    return args


data = []
args = get_arguments()
os.chdir(args.results_dir)
os.chdir("thists")
working_dir = os.getcwd()
pattern = r"0\.\d*"
regex = re.compile(pattern)
for directory in glob("cut_0.*"):
    print("On directory:", directory)
    cut = regex.findall(directory)[0]
    os.chdir(directory)
    shutil.copy(os.path.join(BASE_DIR, DATACARD), "./")
    returncode = os.system(COMBINE_CMD)
    if returncode == 0:
        pass
    else:
        print("Oh no!")
        continue
    with uproot.open(COMBINE_OUTNAME) as f:
        sig = f["limit"]["limit"].arrays()["limit"][0]
        print("\tPrevious sig.:", sig)
    data.append((cut, sig))
    os.chdir(working_dir)
