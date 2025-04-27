import copy
import law
import luigi
import math
import os
import tempfile

from FLAF.RunKit.run_tools import natural_sort
from FLAF.RunKit.crabLaw import update_kinit
from FLAF.RunKit.law_wlcg import WLCGFileTarget
from FLAF.Common.Setup import Setup

class Task(law.Task):
    """
    Base task that we use to force a version parameter on all inheriting tasks, and that provides
    some convenience methods to create local file and directory targets at the default data path.
    """
    version = luigi.Parameter()
    prefer_params_cli = [ 'version' ]
    period = luigi.Parameter()
    customisations =luigi.Parameter(default="")
    test = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        super(Task, self).__init__(*args, **kwargs)
        self.setup = Setup.getGlobal(os.getenv("ANALYSIS_PATH"), self.period, self.customisations)

    def store_parts(self):
        return (self.__class__.__name__, self.version, self.period)

    @property
    def cmssw_env(self):
        return self.setup.cmssw_env

    @property
    def samples(self):
        return self.setup.samples

    @property
    def global_params(self):
        return self.setup.global_params

    @property
    def fs_nanoAOD(self):
        return self.setup.get_fs('nanoAOD')

    @property
    def fs_anaCache(self):
        return self.setup.get_fs('anaCache')

    @property
    def fs_anaTuple(self):
        return self.setup.get_fs('anaTuple')

    # @property
    # def fs_anaCacheTuple(self):
    #     return self.setup.get_fs('anaCacheTuple')

    # @property
    # def fs_nnCacheTuple(self):
    #     return self.setup.get_fs('nnCacheTuple')
    @property
    def fs_NNInputTuple(self):
        return self.setup.get_fs('NNInputTuple')


    @property
    def fs_histograms(self):
        return self.setup.get_fs('histograms')

    def ana_path(self):
        return os.getenv("ANALYSIS_PATH")

    def ana_data_path(self):
        return os.getenv("ANALYSIS_DATA_PATH")

    def local_path(self, *path):
        parts = (self.ana_data_path(),) + self.store_parts() + path
        return os.path.join(*parts)

    def local_target(self, *path):
        return law.LocalFileTarget(self.local_path(*path))

    def remote_target(self, *path, fs=None):
        fs = fs or self.setup.fs_default
        path = os.path.join(*path)
        if type(fs) == str:
            path = os.path.join(fs, path)
            return law.LocalFileTarget(path)
        return WLCGFileTarget(path, fs)

    def law_job_home(self):
        if 'LAW_JOB_HOME' in os.environ:
            return os.environ['LAW_JOB_HOME'], False
        os.makedirs(self.local_path(), exist_ok=True)
        return tempfile.mkdtemp(dir=self.local_path()), True

    def poll_callback(self, poll_data):
        update_kinit(verbose=0)

    def iter_samples(self):
        for sample_id, sample_name in enumerate(natural_sort(self.samples.keys())):
            yield sample_id, sample_name


