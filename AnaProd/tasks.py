import law
import os
import yaml
import contextlib
import luigi
import threading


from FLAF.RunKit.run_tools import ps_call
from FLAF.RunKit.crabLaw import cond as kInit_cond, update_kinit_thread
from FLAF.run_tools.law_customizations import HTCondorWorkflow, copy_param,get_param_value
from run_tools.law_customizations import Task
from FLAF.AnaProd.tasks import AnaTupleTask, DataMergeTask, AnaCacheTupleTask, DataCacheMergeTask, AnaCacheTask

import importlib

class NNInputTupleTask(Task, HTCondorWorkflow, law.LocalWorkflow):

    max_runtime = copy_param(HTCondorWorkflow.max_runtime, 30.0)
    n_cpus = copy_param(HTCondorWorkflow.n_cpus, 1)
    '''
    def workflow_requires(self):
        branches_set = set()
        for branch_idx, (sample_name, sample_type, input_file, ana_br_idx, spin, mass) in self.branch_map.items():
            if ana_br_idx not in branches_set:
                branches_set.add(ana_br_idx)
        return { "anaTuple" :AnaTupleTask.req(self, branches=tuple(branches_set),customisations=self.customisations)}

    def requires(self):
        sample_name, sample_type, input_file, ana_br_idx, spin, mass =  self.branch_data
        return [
            AnaTupleTask.req(self, max_runtime=AnaTupleTask.max_runtime._default, branch=ana_br_idx, branches=(ana_br_idx,),customisations=self.customisations)
        ]
    '''
    def workflow_requires(self):
        workflow_dict = {}
        workflow_dict["anaTuple"] = {
            br_idx: AnaTupleTask.req(self, branch=br_idx)
            for br_idx, _ in self.branch_map.items()
        }
        return workflow_dict

    def requires(self):
        return [ AnaTupleTask.req(self, max_runtime=AnaTupleTask.max_runtime._default) ]

    def create_branch_map(self):
        branches = {}
        anaProd_branch_map = AnaTupleTask.req(self, branch=-1, branches=()).branch_map
        for br_idx, (sample_id, sample_name, sample_type, input_file) in anaProd_branch_map.items():
            branches[br_idx] = (sample_name, sample_type)
        return branches

    def output(self):
        sample_name, sample_type = self.branch_data
        outFileName = os.path.basename(self.input()[0].path)
        output_path = os.path.join('NNInputTuples', self.period, sample_name,self.version, outFileName)#self.version, self.period, sample_name, outFileName)
        return self.remote_target(output_path, fs=self.fs_NNInputTuple)

    def run(self):
        sample_name, sample_type = self.branch_data
        unc_config = os.path.join(self.ana_path(), 'config', self.period, f'weights.yaml')
        producer_nnInputTuples = os.path.join(self.ana_path(), 'AnaProd', 'NNInputTupleProducer.py')
        global_config = os.path.join(self.ana_path(), 'config', 'global.yaml')
        thread = threading.Thread(target=update_kinit_thread)
        thread.start()
        try:
            job_home, remove_job_home = self.law_job_home()
            input_file = self.input()[0]
            print(f"considering sample {sample_name}, {sample_type} and file {input_file.path}")
            with self.input()[0].localize("r") as local_input, self.output().localize("w") as out_file:
                nnInputProducer_cmd = [
                    "python3", producer_nnInputTuples,
                    "--inFileName", local_input.path,
                    "--outFileName", out_file.path,
                    "--uncConfig", unc_config,
                    "--globalConfig", global_config,
                    "--period", self.period,
                ]
                if self.global_params['store_noncentral'] and sample_type != 'data':
                    nnInputProducer_cmd.extend(['--compute_unc_variations', 'True'])
                ps_call(nnInputProducer_cmd, env=self.cmssw_env, verbose=1)
            print(f"finished to produce NNInputTuples for sample {sample_name}")

        finally:
            kInit_cond.acquire()
            kInit_cond.notify_all()
            kInit_cond.release()
            thread.join()







