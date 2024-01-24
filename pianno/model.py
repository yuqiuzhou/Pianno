import nni
import scanpy as sc
import numpy as np
import pandas as pd
import json
import h5py
import sys
sys.path.append("../")
from _spatial_pattern_recognition import spatial_pattern_recognition
import os
from os.path import join, exists
# data
cwd = os.getcwd()
h5py.get_config().track_order = True
adata = sc.read_h5ad(join(cwd, "nni_experiment", "adata.h5ad"))
cfd = adata.uns['cfd']
if exists(join(cfd, "initial_pattern.json")):
  with open(join(cfd, "initial_pattern.json"),'r') as f:
    initial_pattern = json.load(f)
else:
  raise ValueError("The initial_pattern.json is missing!") 
MarkerList = []
ValueType = []
for k, i in initial_pattern.items():
    if "_" in k:
        continue
    elif len(i) == 0:
        continue
    else:
        ValueType.append(k)
        MarkerList.extend(i)
# %%
params = {'n_class': 3,
  'dilation_radius': 1,
  'denoise_weight': 0.001,
  'unsharp_radius': 2,
  'unsharp_amount': 4,
  'gaussian_blur': 1}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)
with open(join(cfd, "best_params.json"),'r') as f:
    best_params = json.load(f)
# %%
print(adata.uns['Patterndict'])
adata = spatial_pattern_recognition(adata, 
                Patterndict=adata.uns['Patterndict'],
                n_class=params["n_class"], 
                unsharp_amount=params["unsharp_amount"],
                dilation_radius=params["dilation_radius"],
                denoise_weight=params["denoise_weight"],
                unsharp_radius=params["unsharp_radius"],
                gaussian_blur=params["gaussian_blur"])
mat = adata[adata.obs["InitialLabel"].isin(ValueType),MarkerList].copy()
df = pd.DataFrame(mat.layers['DenoisedX'],columns=MarkerList,index=mat.obs.index)
df['InitialLabel'] = mat.obs['InitialLabel']
df = df.groupby('InitialLabel').mean()
print(1)
print(df)
weight = 0
for vt in ValueType:
    if vt in df.index:
        continue
    else:
        df.loc[vt]=[0]*len(MarkerList)
        weight += 10   
df = (df - df.min())/(df.max() - df.min())
print(2)
print(df)
df = df.loc[ValueType, MarkerList]
print(3)
print(df)
df = df*(df > 0.1)
print(4)
print(df)
dist = np.sum(np.sum((df - np.identity(len(MarkerList)))**2)) + weight
nni.report_final_result(dist)
# %%
if best_params != {}:
    last_dist = float(list(best_params.keys())[0])
    if last_dist > dist:
        best_params = {}
        best_params[dist] = params
else:
    best_params[dist] = params
with open(join(cfd, "best_params.json"),"w") as f:
  json.dump(best_params,f, indent=4, ensure_ascii=False)