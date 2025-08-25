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
from pprint import pprint

from Studies.DNN.model_generation.parse_column_names import parse_column_names
import FLAF.Common.Utilities as Utilities
import Analysis.H_mumu as analysis


class DNNProducer:


    def __init__(self, cfg, payload_name):

        print("************** Init DNNProducer")

        # cfg is H_mumu/configs/global.yaml
        self.cfg = cfg
        self.global_cfg = self._load_global_config()
        self.payload_name = payload_name
        self.period = "Run3_2022"
        self._load_framework()
        self._set_environ_vars()
        self.parity, self.input_features = self._load_dnn_config()
        self.corrected_input_features = [f"{x}_renorm" for x in self.input_features]
        self.vars_to_save = [f"{self.payload_name}_{col}" for col in self.corrected_input_features]
        self.cols_to_save = [f"{self.payload_name}_{col}" for col in self.input_features]
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
        for i in range(self.parity):
            filename = os.path.join(directory, f"trained_model_{i}.onnx")
            model = ort.InferenceSession(filename)
        return models


    def _load_dnn_config(self):
        """
        Read in the config used to train the network and input features used
        """
        directory = os.path.join(os.environ["ANALYSIS_PATH"], "Studies", "DNN")
        filepath = os.path.join(directory, "configs", "kfold_config.toml")
        with open(filepath, 'rb') as f:
            config = tomllib.load(f)
        pairity = config['kfold']['k']
        # Input features
        filepath = os.path.join(directory, "ds_setup", "general.yaml")
        with open(filepath, "r") as f:
            columns_config = yaml.safe_load(f)
        input_features = parse_column_names(columns_config['vars_to_save'], column_type="data")
        return pairity, input_features


    def _load_framework(self):
        """
        Load any needed files from FLAF
        """
        sys.path.append(os.environ['ANALYSIS_PATH'])
        ROOT.gROOT.ProcessLine(".include "+ os.environ['ANALYSIS_PATH'])
        ROOT.gInterpreter.Declare(f'#include "FLAF/include/Utilities.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/HistHelper.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/AnalysisTools.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/AnalysisMath.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/MT2.h"')
        ROOT.gROOT.ProcessLine(f'#include "FLAF/include/Lester_mt2_bisect.cpp"')

    ### Functions for running the inference ###

    def prepare_dfw(self, dfw):
        dfw = analysis.DataFrameBuilderForHistograms(dfw.df, self.global_cfg, self.period)
        dfw = analysis.PrepareDfForHistograms(dfw)
        # This is probably where I need to do the input renorming?
        for col in self.input_features:
            m = dfw.Mean(col).GetValue()
            s = dfw.StdDev(col).GetValues()
            dfw.Define("{col}_renorm", f"({col} - {m})/{s}")
        return dfw


    def ApplyDNN(self, branches):
        print("*********** Running ApplyDNN...")
        nEvents = len(branches)
        print(f"Running DNN Over {nEvents} events")

        all_predictions = np.zeros(nEvents)
        event_number = np.array(getattr(branches, 'FullEventId'))
        input_array = np.array([getattr(branches, feature_name) for feature_name in self.corrected_input_features]).transpose()
        all_predictions = np.zeros(nEvents, nParity)
        for parityIdx, sess in enumerate(models):

            predictions = sess.run(input_array) 
            # Mask entries that shouldn't be evaluated by this model to 0
            predictions = np.where(
                    event_number % self.parity != parityIdx,
                    predictions,
                    0.0
                )
            all_predictions[parityIdx,:] = predictions

        final_predictions = np.sum(all_predictions, axis=2)

        # Last save the branches
        branches['NNOutput'] = final_predictions.transpose()[class_idx].astype(np.float32)

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
            if col not in self.vars_to_save and col not in self.cols_to_save:
                if col != 'FullEventId':
                    del array[col]
        # Rename the branches
        for col in self.cfg['columns']:
            if col in array.fields:
                array[f"{self.payload_name}_{col}"] = array[f"{col}"]
                del array[f"{col}"]
            else:
                print(f"Expected column {col} not found in your payload array!")
        return array
