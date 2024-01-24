import scanpy as sc
import json
import os
from os.path import join, exists
from pianno._search_best_params import SearchBestParams
from pianno._spatial_pattern_recognition import SpatialPatternRecognition

def _select_top_marker(adata, groupby='InitialLabel',top_n=10, min_lfc=1):
    df_markers = sc.get.rank_genes_groups_df(adata, 
                key="rank_genes_groups",
                group=adata.obs[groupby].unique())
    df_markers = df_markers.loc[~ df_markers.names.isna()]
    df_markers = df_markers[df_markers['logfoldchanges'] > min_lfc]
    df_markers['abs_score']=df_markers.scores.abs()
    df_markers.sort_values('abs_score',ascending=False,inplace=True)
    topN_markers = df_markers.groupby('group').head(top_n).sort_values('group')
    Patterndict={}
    patterns = topN_markers['group'].unique()
    for pt in patterns:
        markers = topN_markers[topN_markers['group'] == pt]['names'].unique().tolist()
        Patterndict[pt] = markers
    return Patterndict

def AutoPatternRecognition(
    adata,
    Patterndict,
    config_path=None,
    param_tuning=True,
    search_space=None,
    port=8080, 
    max_trial_number=100,
    max_experiment_duration='10m',
    small_objects=2
):
    if not config_path:
        config_path = os.getcwd()
    else:
        if not exists(config_path):
            os.mkdir(config_path)
    
    # marker list    
    if isinstance(Patterndict, dict):
        if not exists(join(config_path, "initial_pattern.json")):
            with open(join(config_path, "initial_pattern.json"),"w") as f:
                json.dump(Patterndict, f, indent=4, ensure_ascii=False)
        else:
            pass
    elif isinstance(Patterndict, int):
        top_n = Patterndict
        Patterndict = _select_top_marker(adata, top_n=top_n)
    else:
        raise ValueError("The pattern list must be specificed!")
       
    if param_tuning:
        best_params = SearchBestParams(adata, 
                    pattern_dict=Patterndict.copy(), 
                    cfd=config_path, 
                    search_space=search_space,
                    port=port,
                    max_trial_number=max_trial_number,
                    max_experiment_duration=max_experiment_duration)
    else:
        if exists(join(config_path, "best_params.json")):
            with open(join(config_path, "best_params.json"),'r') as f:
                best_params_dict = json.load(f)
            for key in best_params_dict:
                best_params = best_params_dict[key]               
        else:
            raise ValueError("best_params.json is not found. set 'param_tuning=True'")
        
        
    adata = SpatialPatternRecognition(adata, 
                Patterndict=Patterndict,
                n_class=best_params["n_class"], 
                unsharp_amount=best_params["unsharp_amount"],
                dilation_radius=best_params["dilation_radius"],
                denoise_weight=best_params["denoise_weight"],
                unsharp_radius=best_params["unsharp_radius"],
                gaussian_blur=best_params["gaussian_blur"],
                small_objects=small_objects)
    adata.uns['Patterndict'] = Patterndict
    
    return adata

def ProposedPatterndict(adata, groupby='InitialLabel', 
                        method='wilcoxon',
                        layer='DenoisedX',
                        top_n=10, min_lfc=1):
    adata = adata[adata.obs[groupby] == adata.obs[groupby]].copy()
    sc.tl.rank_genes_groups(adata, groupby, 
                            method=method,
                            layer=layer)
    Patterndict = _select_top_marker(adata, groupby=groupby,
                                     top_n=top_n, min_lfc=min_lfc)
    Patterndict
    return Patterndict