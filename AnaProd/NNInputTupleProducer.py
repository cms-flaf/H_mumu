import os
import sys
import yaml
import ROOT
import datetime
import time
import shutil

ROOT.EnableThreadSafety()

from FLAF.Common.Utilities import DeclareHeader
from FLAF.RunKit.run_tools import ps_call
import FLAF.Common.Utilities as Utilities
import Analysis.H_mumu as analysis

scales = ['Up','Down']

def getKeyNames(root_file_name):
    root_file = ROOT.TFile(root_file_name, "READ")
    key_names = [str(k.GetName()) for k in root_file.GetListOfKeys() ]
    root_file.Close()
    return key_names


def creatNNInputTuple(inFileName, outFileName, unc_cfg_dict, global_cfg_dict, period, snapshotOptions, compute_unc_variations):
    start_time = datetime.datetime.now()
    verbosity = ROOT.Experimental.RLogScopedVerbosity(ROOT.Detail.RDF.RDFLogChannel(), ROOT.Experimental.ELogLevel.kInfo)
    snaps = []
    all_files = []
    kwargset = {}
    kwargset['colToSave'] = []
    file_keys = getKeyNames(inFileName)
    dfw = analysis.DataFrameBuilderForHistograms(ROOT.RDataFrame('Events',inFileName),global_cfg_dict, period, **kwargset)
    analysis.PrepareDfForNNInputs(dfw)
    # varToSave = Utilities.ListToVector(dfw.colToSave)
    all_files.append(f'{outFileName}_Central.root')
    snaps.append(dfw.df.Snapshot(f"Events", f'{outFileName}_Central.root', dfw.colToSave, snapshotOptions))
    print("append the central snapshot")
    '''
    if compute_unc_variations:
        dfWrapped_central = Utilities.DataFrameBuilderBase(df_begin)
        colNames =  dfWrapped_central.colNames
        colTypes =  dfWrapped_central.colTypes
        dfWrapped_central.df = createCentralQuantities(df_begin, colTypes, colNames)
        if dfWrapped_central.df.Filter("map_placeholder > 0").Count().GetValue() <= 0 : raise RuntimeError("no events passed map placeolder")
        print("finished defining central quantities")
        snapshotOptions.fLazy=False
        for uncName in unc_cfg_dict['shape']:
            for scale in scales:
                treeName = f"Events_{uncName}{scale}"
                treeName_noDiff = f"{treeName}_noDiff"
                if treeName_noDiff in file_keys:
                    df_noDiff = ROOT.RDataFrame(treeName_noDiff, inFileName)
                    dfWrapped_noDiff = Utilities.DataFrameBuilderBase(df_noDiff)
                    dfWrapped_noDiff.CreateFromDelta(colNames, colTypes)
                    dfWrapped_noDiff.AddMissingColumns(colNames, colTypes)
                    dfW_noDiff = Utilities.DataFrameWrapper(dfWrapped_noDiff.df,defaultColToSave)
                    applyLegacyVariables(dfW_noDiff,global_cfg_dict, False)
                    varToSave = Utilities.ListToVector(dfW_noDiff.colToSave)
                    all_files.append(f'{outFileName}_{uncName}{scale}_noDiff.root')
                    dfW_noDiff.df.Snapshot(treeName_noDiff, f'{outFileName}_{uncName}{scale}_noDiff.root', varToSave, snapshotOptions)
                treeName_Valid = f"{treeName}_Valid"
                if treeName_Valid in file_keys:
                    df_Valid = ROOT.RDataFrame(treeName_Valid, inFileName)
                    dfWrapped_Valid = Utilities.DataFrameBuilderBase(df_Valid)
                    dfWrapped_Valid.CreateFromDelta(colNames, colTypes)
                    dfWrapped_Valid.AddMissingColumns(colNames, colTypes)
                    dfW_Valid = Utilities.DataFrameWrapper(dfWrapped_Valid.df,defaultColToSave)
                    applyLegacyVariables(dfW_Valid,global_cfg_dict, False)
                    varToSave = Utilities.ListToVector(dfW_Valid.colToSave)
                    all_files.append(f'{outFileName}_{uncName}{scale}_Valid.root')
                    dfW_Valid.df.Snapshot(treeName_Valid, f'{outFileName}_{uncName}{scale}_Valid.root', varToSave, snapshotOptions)
                treeName_nonValid = f"{treeName}_nonValid"
                if treeName_nonValid in file_keys:
                    df_nonValid = ROOT.RDataFrame(treeName_nonValid, inFileName)
                    dfWrapped_nonValid = Utilities.DataFrameBuilderBase(df_nonValid)
                    dfWrapped_nonValid.AddMissingColumns(colNames, colTypes)
                    dfW_nonValid = Utilities.DataFrameWrapper(dfWrapped_nonValid.df,defaultColToSave)
                    applyLegacyVariables(dfW_nonValid,global_cfg_dict, False)
                    varToSave = Utilities.ListToVector(dfW_nonValid.colToSave)
                    all_files.append(f'{outFileName}_{uncName}{scale}_nonValid.root')
                    dfW_nonValid.df.Snapshot(treeName_nonValid, f'{outFileName}_{uncName}{scale}_nonValid.root', varToSave, snapshotOptions)
    '''
    print(f"snaps len is {len(snaps)}")
    snapshotOptions.fLazy = True
    if snapshotOptions.fLazy == True:
        print("going to rungraph")
        ROOT.RDF.RunGraphs(snaps)
    return all_files


if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--inFileName', required=True, type=str)
    parser.add_argument('--outFileName', required=True, type=str)
    parser.add_argument('--period', required=True, type=str)
    parser.add_argument('--uncConfig', required=True, type=str)
    parser.add_argument('--globalConfig', required=True, type=str)
    parser.add_argument('--compute_unc_variations', type=bool, default=False)
    parser.add_argument('--compressionLevel', type=int, default=4)
    parser.add_argument('--compressionAlgo', type=str, default="ZLIB")
    args = parser.parse_args()
    for header in [ "FLAF/include/HistHelper.h", "FLAF/include/Utilities.h","include/Helper.h", "include/HmumuCore.h", "FLAF/include/AnalysisTools.h" ]:
        DeclareHeader(os.environ["ANALYSIS_PATH"]+"/"+header)


    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    snapshotOptions.fOverwriteIfExists=True
    snapshotOptions.fLazy = True
    snapshotOptions.fMode="RECREATE"
    snapshotOptions.fCompressionAlgorithm = getattr(ROOT.ROOT, 'k' + args.compressionAlgo)
    snapshotOptions.fCompressionLevel = args.compressionLevel
    unc_cfg_dict = {}

    with open(args.uncConfig, 'r') as f:
        unc_cfg_dict = yaml.safe_load(f)

    global_cfg_dict = {}
    with open(args.globalConfig, 'r') as f:
        global_cfg_dict = yaml.safe_load(f)

    startTime = time.time()
    outFileNameFinal = f'{args.outFileName}'


    all_files = creatNNInputTuple(args.inFileName, args.outFileName.split('.')[0], unc_cfg_dict, global_cfg_dict, args.period, snapshotOptions, args.compute_unc_variations)

    hadd_str = f'hadd -f209 -n10 {outFileNameFinal} '
    hadd_str += ' '.join(f for f in all_files)
    if len(all_files) > 1:
        ps_call([hadd_str], True)
    else:
        shutil.copy(all_files[0],outFileNameFinal)
    if os.path.exists(outFileNameFinal):
            for histFile in all_files:
                if histFile == outFileNameFinal: continue
                os.remove(histFile)
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
