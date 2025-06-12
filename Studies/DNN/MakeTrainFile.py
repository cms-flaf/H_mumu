import argparse
from glob import glob
import os
import sys

import Analysis.H_mumu as analysis
import FLAF.Common.Utilities as Utilities
import psutil
import ROOT
import yaml

from model_generation.parse_column_names import parse_column_names

ROOT.gROOT.SetBatch(True)
ROOT.EnableThreadSafety()
from FLAF.Common.Utilities import DeclareHeader



def create_file(
    config_dict, global_cfg_dict, general_cfg_dict, period, output_folder, out_filename
):
    print(
        f"Starting create file. Memory usage in MB is {get_memory_usage()}"
    )
    nBatches = config_dict['meta_data']['batch_dict']['nBatches']
    print(config_dict.keys())
    #for process in config_dict["processes"]:
    #    if (nBatches is None) or (
    #        (process["nBatches"] <= nBatches) and (process["nBatches"] != 0)
    #    ):
    #        nBatches = process["nBatches"]

    print(f"Going to make {nBatches} batches")
    batch_size = config_dict["meta_data"]["batch_dict"]["batch_size"]

    step_idx = 0

    # Get the name/type (And order!) of signal columns
    master_column_names = []
    master_column_types = []
    master_column_names_vec = ROOT.std.vector("string")()
    # Assume master(signal) is saved first and use idx==0 entry to fill

    for process in config_dict["processes"]:
        process_filelist = [f"{x}/*.root" for x in process["datasets"]]

        tmp_dir = config_dict['meta_data']['temp_folder']
        tmp_filename = os.path.join(tmp_dir, f"tmp{step_idx}.root")
        tmpnext_filename = os.path.join(tmp_dir, f"tmp{step_idx+1}.root")

        # print(process_filelist)
        df_in = ROOT.RDataFrame("Events", process_filelist)

        # Filter for nLeps and Parity (iterate cut in config)
        if 'iterate_cut' in config_dict["meta_data"].keys():
            df_in = df_in.Filter(config_dict["meta_data"]["iterate_cut"])

        nEntriesPerBatch = process["batch_size"]
        nBatchStart = process["batch_start"]
        nBatchEnd = nBatchStart + nEntriesPerBatch

        # Load df_out, if first iter then load an empty, otherwise load the past file
        if step_idx == 0:
            df_out = ROOT.RDataFrame(nBatches * batch_size)
            df_out = df_out.Define("is_valid", "false")

            # Fill master column nametype
            master_column_names = df_in.GetColumnNames()
            master_column_types = [
                str(df_in.GetColumnType(str(c))) for c in master_column_names
            ]
            for name in master_column_names:
                master_column_names_vec.push_back(name)
        else:
            df_out = ROOT.RDataFrame("Events", tmp_filename)

        local_column_names = df_in.GetColumnNames()
        local_column_types = [
            str(df_in.GetColumnType(str(c))) for c in local_column_names
        ]
        local_column_names_vec = ROOT.std.vector("string")()
        for name in local_column_names:
            local_column_names_vec.push_back(name)

        # Need a local_to_master_map so that local columns keep the same index as the master columns
        local_to_master_map = [
            list(master_column_names).index(local_name)
            for local_name in local_column_names
        ]
        master_size = len(master_column_names)

        queue_size = 10
        max_entries = nEntriesPerBatch * nBatches

        tuple_maker = ROOT.analysis.TupleMaker(*local_column_types)(
            queue_size, max_entries
        )

        df_out = tuple_maker.FillDF(
            ROOT.RDF.AsRNode(df_out),
            ROOT.RDF.AsRNode(df_in),
            local_to_master_map,
            master_size,
            local_column_names_vec,
            nBatchStart,
            nBatchEnd,
            batch_size,
        )

        for column_idx, column_name in enumerate(master_column_names):
            column_type = master_column_types[column_idx]

            if step_idx == 0:
                df_out = df_out.Define(
                    str(column_name),
                    f"_entry ? _entry->GetValue<{column_type}>({column_idx}) : {column_type}() ",
                )
            else:
                if column_name not in local_column_names:
                    continue
                df_out = df_out.Redefine(
                    str(column_name),
                    f"_entry ? _entry->GetValue<{column_type}>({column_idx}) : {column_name} ",
                )

        df_out = df_out.Redefine("is_valid", "(is_valid) || (_entry)")

        snapshotOptions = ROOT.RDF.RSnapshotOptions()
        # snapshotOptions.fOverwriteIfExists=False
        # snapshotOptions.fLazy=True
        snapshotOptions.fMode = "RECREATE"
        snapshotOptions.fCompressionAlgorithm = getattr(ROOT.ROOT, "k" + "ZLIB")
        snapshotOptions.fCompressionLevel = 4
        ROOT.RDF.Experimental.AddProgressBar(df_out)
        print("Going to snapshot")
        save_column_names = ROOT.std.vector("string")(master_column_names)
        save_column_names.push_back("is_valid")
        df_out.Snapshot("Events", tmpnext_filename, save_column_names, snapshotOptions)

        tuple_maker.join()

        step_idx += 1

        os.system(f"rm {tmp_filename}")

    print("Finished create file loop, now we must add the DNN variables")
    # Increment the name indexes before I embarass myself again
    tmp_dir = config_dict['meta_data']['temp_folder']
    tmp_filename = os.path.join(tmp_dir, f"tmp{step_idx}.root")
    tmpnext_filename = os.path.join(tmp_dir, f"tmp{step_idx+1}.root")

    df_out = ROOT.RDataFrame("Events", tmp_filename)

    snapshotOptions = ROOT.RDF.RSnapshotOptions()
    # snapshotOptions.fOverwriteIfExists=False
    # snapshotOptions.fLazy=True
    snapshotOptions.fMode = "RECREATE"
    snapshotOptions.fCompressionAlgorithm = getattr(ROOT.ROOT, "k" + "ZLIB")
    snapshotOptions.fCompressionLevel = 4
    ROOT.RDF.Experimental.AddProgressBar(df_out)
    print("Going to snapshot")
    # Only need to save the prexisting columns plus the new DNN variables
    # to add kwargset for isData
    dfw_out = analysis.DataFrameBuilderForHistograms(df_out, global_cfg_dict, period)
    dfw_out = analysis.PrepareDfForNNInputs(dfw_out)
    dfw_out.colToSave += [c for c in df_out.GetColumnNames()]
    col_to_save = parse_column_names(general_cfg_dict['vars_to_save'],  column_type='all')

    dfw_out.df.Snapshot(
        "Events", tmpnext_filename, Utilities.ListToVector(col_to_save), snapshotOptions
    )

    print(f"Finished create file, will copy tmp file to final output {out_filename}")
    print(f"cp {tmpnext_filename} {out_filename}")
    print(f"rm {tmp_filename}")
    print(f"rm {tmpnext_filename}")
    os.system(f"cp {tmpnext_filename} {out_filename}")
    os.system(f"rm {tmp_filename}")
    os.system(f"rm {tmpnext_filename}")

def set_environ_vars():
    sys.path.append(os.environ["ANALYSIS_PATH"])
    ana_path = os.environ["ANALYSIS_PATH"]
    for header in [
        "FLAF/include/Utilities.h",
        "include/Helper.h",
        "include/HmumuCore.h",
        "FLAF/include/AnalysisTools.h",
        "FLAF/include/AnalysisMath.h",
    ]:
        DeclareHeader(os.environ["ANALYSIS_PATH"] + "/" + header)

def get_args():
    parser = argparse.ArgumentParser(description="Create train/test file(s) for DNN.")
    parser.add_argument(
        "--config-folder", required=True, type=str, help="Config Folder containing batch_config_x.yaml file(s)"
    )
    parser.add_argument("--period", required=True, type=str, help="period")
    args = parser.parse_args()
    return args

def load_headers():
    headers_dir = os.path.dirname(os.path.abspath(__file__))
    # headers = [ 'AnalysisTools.h', 'TupleMaker.h' ] #Order here matters since TupleMaker requires AnalysisTools
    headers = [
        "include/TupleMaker.h"
    ]  # Order here matters since TupleMaker requires AnalysisTools
    for header in headers:
        header_path = os.path.join(headers_dir, header)
        if not ROOT.gInterpreter.Declare(f'#include "{header_path}"'):
            raise RuntimeError(f"Failed to load {header_path}")

def get_memory_usage():
    return psutil.Process(os.getpid()).memory_info()[0] / float(2 ** 20)

def yaml_to_config(yamlname):

    with open(yamlname, "r") as file:
        config_dict = yaml.safe_load(file)
    glb_cfg_dict_name = config_dict["meta_data"]["global_config"]
    with open(glb_cfg_dict_name, "r") as glb_cfg_file:
        global_cfg_dict = yaml.safe_load(glb_cfg_file)
    general_cfg_dict_name = config_dict["meta_data"]["general_config"]
    with open(general_cfg_dict_name, "r") as general_cfg_dict_file:
        general_cfg_dict = yaml.safe_load(general_cfg_dict_file)

    return config_dict, global_cfg_dict, general_cfg_dict

if __name__ == "__main__":

    set_environ_vars()
    args = get_args()
    load_headers()

    for yamlname in glob(os.path.join(args.config_folder, "batch_config*.yaml")):
        config_dict, global_cfg_dict, general_cfg_dict = yaml_to_config(yamlname)
        create_file(
            config_dict,
            global_cfg_dict,
            general_cfg_dict,
            args.period,
            args.config_folder,
            f"{config_dict['meta_data']['output_folder']}/{config_dict['meta_data']['output_name']}"
        )
