import os
import shutil
from glob import glob

import uproot

### Constants
COMBINE_CMD = "combine -M Significance datacard.txt -t -1 --expectSignal=1"
COMBINE_OUTNAME = "higgsCombineTest.Significance.mH120.root"
BASE_DIR = "/afs/cern.ch/user/a/ayeagle/H_mumu/Studies/DNN/"
DATACARD = BASE_DIR + "datacard.txt"


for directory in glob(BASE_DIR + "results/*"):
    print("On directory:", directory)
    previous_path = directory + "/" + COMBINE_OUTNAME
    root_path = directory + "/" + "hists.root"
    if os.path.isfile(previous_path):
        print("\tAlready did this one!")
        with uproot.open(previous_path) as f:
            sig = f["limit"]["limit"].arrays()["limit"][0]
            print("\tPrevious sig.:", sig)
    else:
        if not os.path.isfile(root_path):
            print("Not THists made. Skipping...")
            continue
        shutil.copy(DATACARD, directory)
        os.chdir(directory)
        returncode = os.system(COMBINE_CMD)
        if returncode == 0:
            pass
        else:
            print("Oh no!")
        os.chdir(BASE_DIR)
