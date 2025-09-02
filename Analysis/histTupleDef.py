
import importlib
from FLAF.Common.Utilities import *
from FLAF.Common.HistHelper import *

if __name__ == "__main__":
    sys.path.append(os.environ['ANALYSIS_PATH'])


defaultColToSave = ["FullEventId","luminosityBlock", "run","event", "sample_type", "period", "isData","PuppiMET_pt", "PuppiMET_phi", "nJet","DeepMETResolutionTune_pt", "DeepMETResolutionTune_phi","DeepMETResponseTune_pt", "DeepMETResponseTune_phi","PV_npvs"]
initialized = False
analysis = None

def Initialize():
    global initialized
    if not initialized:
        headers_dir = os.path.dirname(os.path.abspath(__file__))
        ROOT.gROOT.ProcessLine(f".include {os.environ['ANALYSIS_PATH']}")
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/HistHelper.h"')
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/Utilities.h"')
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/pnetSF.h"')
        ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisTools.h"')
        ROOT.gROOT.ProcessLine('#include "FLAF/include/AnalysisMath.h"')
        ROOT.gInterpreter.Declare(f'#include "include/Helper.h"') # not related to FullEvtId definition but needed for analysis specific purpose. At a certain point it will be moved to analysis specific section.
        initialized = True

def analysis_setup(setup):
    global analysis
    analysis_import = (setup.global_params['analysis_import'])
    analysis = importlib.import_module(f'{analysis_import}')




def GetDfw(df, df_cache, global_params, shift="Central", col_names_central=[], col_types_central=[], cache_map_name="cache_map_Central"):
    period = global_params["era"]
    kwargset = {} # here go the customisations for each analysis eventually extrcting stuff from the global params
    kwargset['isData'] = global_params['dataset'] == 'data'
    kwargset['wantTriggerSFErrors'] = global_params['dataset'] != 'data' # already set not in data
    kwargset['colToSave'] = []

    dfw = analysis.DataFrameBuilderForHistograms(df, global_params, period, **kwargset)

    if df_cache:
        dfWrapped_cache = analysis.DataFrameBuilderForHistograms(df_cache, global_params, **kwargset)
        AddCacheColumnsInDf(dfw, dfWrapped_cache, cache_map_name)
    if shift == "Valid":
        dfw.CreateFromDelta(col_names_central, col_types_central)
    if shift != "Central":
        dfw.AddMissingColumns(col_names_central, col_types_central)
    new_dfw = analysis.PrepareDfForHistograms(dfw)
    return new_dfw

def DefineWeightForHistograms(dfw, uncName, uncScale, sample_type, unc_cfg_dict, hist_cfg_dict, global_params, final_weight_name = "weight_for_hists"):
    categories = global_params['categories']
    isCentral = uncName == 'Central'
    total_weight_expression = analysis.GetWeight("muMu", '','') if sample_type!='data' else "1" # are we sure?
    weight_name = "final_weight"
    if weight_name not in dfw.df.GetColumnNames():
        dfw.df = dfw.df.Define(weight_name, total_weight_expression)
    if not isCentral and type(unc_cfg_dict)==dict:
        if uncName in unc_cfg_dict.keys() and 'expression' in unc_cfg_dict[uncName].keys():
            weight_name = unc_cfg_dict[uncName]['expression'].format(scale=uncScale)
    dfw.df = dfw.df.Define(final_weight_name, weight_name)



