from __future__ import annotations
import os, sys
import numpy as np
import awkward as ak
import onnxruntime as ort
import psutil
import yaml
import os
import ROOT
import tomllib
import pickle as pkl

from Studies.DNN.model_generation.parse_column_names import parse_column_names
import FLAF.Common.Utilities as Utilities
import Analysis.H_mumu as analysis
from FLAF.Common.Utilities import DeclareHeader


class DNNProducer:

    def __init__(self, cfg, payload_name):
        # cfg is H_mumu/configs/global.yaml
        self.cfg = cfg
        self.global_cfg = self._load_global_config()
        self.payload_name = payload_name
        self.period = "Run3_2022"
        self._load_framework()
        self._set_environ_vars()
        self.parity, self.input_features = self._load_dnn_config()
        # self.corrected_input_features = [f"{x}_renorm" for x in self.input_features]
        # Columns for tmp file
        self.vars_to_save = self.input_features  # + self.corrected_input_features
        # Final columns
        self.cols_to_save = [
            f"{self.payload_name}_{col}" for col in self.cfg["columns"]
        ]
        self.models = self._load_models()

    ### Init helpers ###

    def _set_environ_vars(self):
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

    def _load_global_config(self):
        filepath = os.path.join(os.environ["ANALYSIS_PATH"], "config", "global.yaml")
        with open(filepath, "r") as f:
            global_config = yaml.safe_load(f)
        return global_config

    def _load_models(self):
        """
        Load in the trained ONNX models
        """
        directory = os.path.join(os.environ["ANALYSIS_PATH"], "Analysis", "models")
        models = []
        renorm_vars = []
        for i in range(self.parity):
            # The model itself
            filename = os.path.join(directory, f"trained_model_{i}.onnx")
            model = ort.InferenceSession(filename)
            models.append(model)
        return models

    def _load_dnn_config(self):
        """
        Read in the config used to train the network and input features used
        """
        directory = os.path.join(os.environ["ANALYSIS_PATH"], "Studies", "DNN")
        filepath = os.path.join(directory, "configs", "config.toml")
        with open(filepath, "rb") as f:
            config = tomllib.load(f)
        pairity = config["kfold"]["k"]
        # Input features
        filepath = os.path.join(directory, "ds_setup", "general.yaml")
        with open(filepath, "r") as f:
            columns_config = yaml.safe_load(f)
        input_features = parse_column_names(
            columns_config["vars_to_save"], column_type="data"
        )
        return pairity, input_features

    def _load_framework(self):
        """
        Load any needed files from FLAF
        """
        sys.path.append(os.environ["ANALYSIS_PATH"])
        ROOT.gROOT.ProcessLine(".include " + os.environ["ANALYSIS_PATH"])
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/Utilities.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/HistHelper.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/AnalysisTools.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/AnalysisMath.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/MT2.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/Lester_mt2_bisect.cpp"')

    ### Functions for running the inference ###

    def prepare_dfw(self, dfw):
        print("*********** Running prepare_dfw...")
        dfw = analysis.DataFrameBuilderForHistograms(
            dfw.df, self.global_cfg, self.period
        )
        dfw = analysis.PrepareDfForHistograms(dfw)
        return dfw

    def ApplyDNN(self, branches):
        print("*********** Running ApplyDNN...")
        nEvents = len(branches)
        event_number = np.array(getattr(branches, "FullEventId"))
        input_array = np.array(
            [getattr(branches, feature_name) for feature_name in self.input_features]
        ).transpose()
        all_predictions = np.zeros([nEvents, self.parity])
        for parityIdx, sess in enumerate(self.models):
            input_name = sess.get_inputs()[0].name
            label_name = sess.get_outputs()[0].name
            print("Input name:", input_name)
            print("Label name:", label_name)
            print("X:", input_array.shape)
            predictions = sess.run([label_name], {input_name: input_array})[0]
            mask = (event_number % self.parity) != parityIdx
            predictions[mask] = 0
            all_predictions[:, parityIdx] = predictions.reshape(predictions.shape[0])
        final_predictions = np.sum(all_predictions, axis=1)
        # Last save the branches
        branches["NNOutput"] = final_predictions.transpose().astype(np.float32)
        print("Finishing call, memory?")
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"Current memory usage: {mem_mb:.2f} MB")
        return branches

    def run(self, array):
        print("############# Running DNN producer")
        array = self.ApplyDNN(array)
        # Delete not-needed branches
        for col in array.fields:
            if col not in self.cfg["columns"]:
                if col != "FullEventId":
                    del array[col]
        # Rename the branches
        for col in self.cfg["columns"]:
            if col in array.fields:
                array[f"{self.payload_name}_{col}"] = array[f"{col}"]
                del array[f"{col}"]
            else:
                print(f"Expected column {col} not found in your payload array!")
        return array
