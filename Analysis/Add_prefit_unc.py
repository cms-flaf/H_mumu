import math
import os
import ROOT
import sys
if __name__ == "__main__":
    sys.path.append(os.environ['ANALYSIS_PATH'])

import FLAF.Common.Utilities as Utilities
from FLAF.Analysis.HistHelper import *
import importlib
import FLAF.Common.Setup as Setup


def addPrefitUncertainties(hist_central,hists_up, hists_down):
    for bin_idx in range(hist_central.GetNbinsX()):
        bin_content = hist_central.GetBinContent(bin_index + 1)
        bin_error = hist_central.GetBinError(bin_index + 1)
        bin_error_squared = bin_error ** 2
        bin_error_up_squared_total = 0.
        bin_error_down_squared_total = 0.
        for hist_up in hists_up:
            bin_content_up = hist_up.GetBinContent(bin_index + 1)
            bin_error_up = hist_up.GetBinError(bin_index + 1)
            bin_error_up_squared_total += bin_error_up ** 2
        for hist_down in hists_down:
            bin_content_down = hist_down.GetBinContent(bin_index + 1)
            bin_error_down = hist_down.GetBinError(bin_index + 1)
            bin_error_down_squared_total += bin_error_down ** 2
        bin_error_varied_tot = 0.5*math.sqrt(bin_error_up_squared_total + bin_error_down_squared_total)
        bin_error_total = math.sqrt(bin_error_squared + bin_error_varied_tot)
        hist_central.SetBinError(bin_index + 1, bin_error_total)


if __name__ == '__main__':
    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--inFile', required=True)
    parser.add_argument('--outFile', required=True)
    # parser.add_argument('--year', required=True)
    # parser.add_argument('--var', required=False, type=str, default='tau1_pt')
    #parser.add_argument('--remove-files', required=False, type=bool, default=False)
    parser.add_argument('--ana_path', required=True, type=str)
    parser.add_argument('--period', required=True, type=str)
    args = parser.parse_args()

    setup = Setup.Setup(args.ana_path, args.period)

    analysis_import = (setup.global_params['analysis_import'])
    analysis = importlib.import_module(f'{analysis_import}')


    samples_to_consider = setup.global_params['sample_types_to_merge']
    if type(samples_to_consider) == list:
        samples_to_consider.append('data')
        for signal_name in setup.signal_samples:
            samples_to_consider.append(signal_name)

    inFile = ROOT.TFile.Open(args.inFile, "READ")
    outFile = ROOT.TFile.Open(args.outFile, "RECREATE")
    channels =[str(key.GetName()) for key in inFile.GetListOfKeys()]

    for channel in channels:
        dir_0 = inFile.Get(channel)
        regions =[str(key.GetName()) for key in dir_0.GetListOfKeys()]

        for region in regions:
            dir_1 = dir_0.Get(region)
            keys_categories = [str(key.GetName()) for key in dir_1.GetListOfKeys()]

            for cat in keys_categories: #This is inclusive/boosted/baseline/res1b/res2b
                dir_2= dir_1.Get(cat)

                for process in samples_to_consider:

                    hists_up = []
                    hists_down = []
                    if process not in dir_2.GetListOfKeys():
                        print(f"Process {process} not found in {cat} for channel {channel}")
                        continue
                    hist_central = dir_2.Get(process)
                    # hist_central = hist_central_key.ReadObj()
                    hist_central.SetDirectory(0)
                    # print(hist_central.GetName(), hist_central.GetTitle())
                    # print(hist_central.GetNbinsX())

                    for unc in setup.weights_config['shape']:
                        hist_name_up = f"{process}_{unc}_{args.year}_Up"
                        if hist_name_up not in dir_2.GetListOfKeys():
                            hist_name_up = f"{process}_{unc}_Up"
                        if hist_name_up not in dir_2.GetListOfKeys():
                            print(f"Process {hist_name_up} not found in {cat} for channel {channel}")
                            continue
                        key_hist_up = dir_2.Get(hist_name_up)
                        hist_up = key_hist_up.ReadObj()
                        hist_up.SetDirectory(0)
                        hists_up.append(hist_up)

                        hist_name_Down = f"{process}_{unc}_{args.year}_Down"
                        if hist_name_Down not in dir_2.GetListOfKeys():
                            hist_name_Down = f"{process}_{unc}_Down"
                        if hist_name_Down not in dir_2.GetListOfKeys():
                            print(f"Process {hist_name_Down} not found in {cat} for channel {channel}")
                            continue
                        key_hist_Down = dir_2.Get(hist_name_Down)
                        hist_Down = key_hist_Down.ReadObj()
                        hist_Down.SetDirectory(0)
                        hists_down.append(hist_Down)

                    addPrefitUncertainties(hist_central,hists_up, hists_down)
                    dirStruct = (channel, cat, region)
                    dir_name = '/'.join(dirStruct)
                    dir_ptr = Utilities.mkdir(outFile,dir_name)
                    obj.SetTitle(process)
                    obj.SetName(process)
                    dir_ptr.WriteTObject(obj, process, "Overwrite")

    outFile.Close()

    '''
    inFile.Close()