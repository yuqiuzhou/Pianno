# %%
from nni.experiment import Experiment
import json
import os
from os.path import join, exists
import shutil
import h5py
# %%
def tuner(adata, cfd, search_space, port=8080, mtr=100, med='10m'):
  tcd = os.path.dirname(__file__)
  ewd = join(tcd, "nni_experiment")
  if not exists(ewd):
    os.mkdir(ewd)
  h5py.get_config().track_order = True
  adata.write_h5ad(join(tcd, "nni_experiment", "adata.h5ad"))
  experiment = Experiment('local')
  experiment.config.trial_command = 'python model.py'
  experiment.config.trial_code_directory = tcd
  experiment.config.search_space = search_space
  experiment.config.tuner.name = 'TPE'
  experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
  experiment.config.max_trial_number = mtr
  experiment.config.trial_concurrency = 2
  experiment.config.max_experiment_duration = med
  experiment.config.experiment_working_directory = ewd
  if exists(join(cfd, "best_params.json")):
    os.remove(join(cfd, "best_params.json"))
  with open(join(cfd, "best_params.json"),"w") as f:
    json.dump({},f, indent=4, ensure_ascii=False)
  experiment.run(int(port))
  shutil.rmtree(ewd)