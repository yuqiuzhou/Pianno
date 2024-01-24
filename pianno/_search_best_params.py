# %%
import json
import os
from os.path import join, exists
from pianno._tuner import tuner
import time
# %%
def SearchBestParams(adata, 
                     pattern_dict, 
                     cfd=None, 
                     search_space=None,
                     port=8080,
                     max_trial_number=100,
                     max_experiment_duration='10m'):
    """
    Args:
        adata (Anndata): An Pianno Object.
        pattern_dict (dict): List of markers of dictionary type.
        cfd (path, optional): Config path. Defaults to None.
        search_space (dict, optional): Search space for Hyperparameter. Defaults to None.
    """   
    # configuration path
    if not cfd:
        cfd = os.getcwd()
    else:
        if not exists(cfd):
            os.mkdir(cfd)
    print("Configuration path of Pianno: {}".format(cfd))
    adata.uns['cfd'] = cfd

    # marker list
    if pattern_dict:
        if not exists(join(cfd, "initial_pattern.json")):
            with open(join(cfd, "initial_pattern.json"),"w") as f:
                json.dump(pattern_dict, f, indent=4, ensure_ascii=False)
        else:
            pass
    else:
        raise ValueError("The pattern dictionary must be specificed!")
    
    adata.uns['Patterndict'] = pattern_dict

    # search space
    if search_space:
        with open(join(cfd, "search_space.json"),"w") as f:
            json.dump(search_space, f, indent=4, ensure_ascii=False)
    else:
        if exists(join(cfd, "search_space.json")):
            with open(join(cfd, "search_space.json"),'r') as f:
                search_space = json.load(f)
        else:
            search_space = {
                "n_class": {"_type": "choice", "_value": [2, 3, 3, 3]},
                "dilation_radius": {"_type": "quniform", "_value": [1,5,1]},
                "denoise_weight": {"_type": "quniform", "_value": [0.01, 0.2, 0.01]},
                "unsharp_radius": {"_type": "quniform", "_value": [1,5,1]},
                "unsharp_amount": {"_type": "quniform", "_value": [1,20,1]},
                "gaussian_blur": {"_type": "quniform", "_value": [1,5,1]}
            }
            with open(join(cfd, "search_space.json"),"w") as f:
                json.dump(search_space, f, indent=4, ensure_ascii=False)
            print("The search_space is undefined so the default configuration will be used.")
            print("You can modified it in {}".format(join(cfd, "search_space.json")))

    tuner(adata, cfd, search_space, 
        port=port, mtr=max_trial_number, 
        med=max_experiment_duration)
    
    with open(join(cfd, "best_params.json"),'r') as f:
        best_params = json.load(f)
        
    return best_params[list(best_params.keys())[0]]