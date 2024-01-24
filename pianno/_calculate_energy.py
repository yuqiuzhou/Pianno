import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing

import scipy
def defaultX(adata,X:str = None):
    if X not in adata.layers.keys():
        raise ValueError("{} is not found in layers!".format(X))
    else:
        if scipy.sparse.issparse(adata.layers[X]):
            adata.X = adata.layers[X].todense()
        else:
            adata.X = adata.layers[X]        
        adata.uns["defaultX"] = X
        return adata

def _local_energy(adata, label_key='InitialLabel', connect_key = 'spatial_connectivities', center_weight=1, LOWER_BOUND=1, feature_range=1):
    InitialLabel = adata.obs[label_key].copy()
    NS, NR = adata.shape[0], len(set(InitialLabel.dropna()))
    InitialLabel.cat.categories = range(NR)
    InitialLabel = pd.DataFrame(InitialLabel)
    spatial_con_bool = adata.obsp[connect_key].A > 0
    
    NeiRegionCount = list(map(lambda x:InitialLabel[x.T].value_counts().sort_index(),spatial_con_bool))
    SpotRegionCount = pd.get_dummies(InitialLabel[label_key]).values * center_weight
    #RegionCount = pd.DataFrame(NeiRegionCount + SpotRegionCount + 1)
    RegionCount = pd.DataFrame(NeiRegionCount + SpotRegionCount + LOWER_BOUND)
    Energy = -np.log(RegionCount.div(RegionCount.sum(1),0))
    #Max-Min normalization
    minmax = preprocessing.MinMaxScaler(feature_range = (0,feature_range))
    Energy = np.round(minmax.fit_transform(Energy),2)
    return Energy, RegionCount

def _global_energy(adata, RegionCount, label_key='InitialLabel', connect_key = 'spatial_connectivities',
                   LOWER_BOUND=0.05, feature_range=1):
    InitialLabel = adata.obs[label_key].copy()
    NS, NR = adata.shape[0], len(set(InitialLabel.dropna()))
    InitialLabel.cat.categories = range(NR)
    TypeCount = np.sum((RegionCount - 1) > 0,1)
    #Label Pair Count
    BoundS = TypeCount[TypeCount > 1].index.values
    spatial_con_bool = adata.obsp[connect_key].A > 0
    edge_con = pd.DataFrame(spatial_con_bool[BoundS][:,BoundS])
    edge_con['CenterLabel'] = InitialLabel[BoundS].values
    CenterLabel = edge_con.groupby('CenterLabel').sum().T
    CenterLabel['NeighborLabel'] = InitialLabel[BoundS].values

    LabelPairCount = CenterLabel.groupby('NeighborLabel').sum()
    LabelPairProb = LabelPairCount.div(LabelPairCount.sum(1),0)
    LabelPairEnergy = -np.log(LabelPairProb+LOWER_BOUND)

    #Feature distance weighted
    feature_image = np.array(adata.uns['FeatureImage'])
    FI = feature_image.reshape(feature_image.shape[0],feature_image.shape[1]*feature_image.shape[2]).T
    FI = FI[adata.obs['spotID'].values,:]
    NeiFI = [FI[x] for x in spatial_con_bool]
    NeiLabel = [InitialLabel[x].to_list() for x in spatial_con_bool]
    # isolated spot
    isolated = list(map(int, np.where(np.array([len(x) for x in NeiFI]) == 0)[0]))
    if len(isolated) != 0:
        for iso in isolated:
            NeiFI[iso] = FI[iso].reshape(1,-1)
            NeiLabel[iso] = [InitialLabel[iso]]    
    euc_dis = [np.round(euclidean_distances(FI[i].reshape(1, -1),NeiFI[i]),2) for i in range(NS)]

    Dss = []
    for rl in range(NR):
        Irr = [np.array([l == rl for l in x]) for i, x in enumerate(NeiLabel)]
        Dss.append([np.mean(Irr[i]*np.exp(euc_dis[i]) + (~Irr[i])*(1 + np.exp(-euc_dis[i]))) for i in range(NS)])
    Dss = np.round(np.array(Dss),2)

    GlobalEnergy = np.array([LabelPairEnergy[InitialLabel[i]] * Dss[:,i] for i in range(NS)])
    #Max-Min normalization
    minmax = preprocessing.MinMaxScaler(feature_range = (0,feature_range))
    GlobalEnergy = np.round(minmax.fit_transform(GlobalEnergy),2)   
    return GlobalEnergy, LabelPairCount

def _find_knn(adata, Patterns=None):
    if Patterns is None:
        Patterns = adata.uns['PatternList'].index
    n_neighs = adata.uns['spatial_neighbors']['params']['n_neighbors']
    Tissue = adata[:,Patterns].copy()
    Tissue = defaultX(Tissue,'DenoisedX')
    sc.pp.scale(Tissue, max_value=10)
    sc.pp.pca(Tissue,use_highly_variable=False)
    sc.pp.neighbors(Tissue,n_neighbors=n_neighs+1)

    connectivities = Tissue.obsp["connectivities"].A
    kth_neigh = np.argsort(-connectivities, axis=1)[:,n_neighs-1]
    kth_connect = [connectivities[i,k] for i,k in enumerate(kth_neigh)]
    knn_connectivities = 1*(connectivities >= kth_connect).T
    adata.obsp["knn_connectivities"] = csr_matrix(knn_connectivities)
    del connectivities, knn_connectivities
    return adata

def CalculateEnergy(adata, label_key='InitialLabel' ,n_rings=1, n_neighs=6, coord_type=None, radius=None,
                    center_weight=1, feature_range=1): 
    n_rings = n_rings
    n_neighs = n_neighs
    coord_type=coord_type
    radius=radius
    if adata.uns['unknown_type'] == 'Background':   
        InitialLabel = adata.obs[label_key].copy()
        InitialLabel[InitialLabel != InitialLabel] = 'Background'
        adata.obs[label_key] = InitialLabel
    print("---Create Spatial Graph:", end=' ')
    sq.gr.spatial_neighbors(adata, n_rings=n_rings, n_neighs=n_neighs, coord_type=coord_type, radius=radius)
    print("Done!")
    center_weight = center_weight
    feature_range = feature_range
    print("---Compute Spatial Energy:", end=' ')
    SpatialEnergy, RegionCount = _local_energy(adata,
                                               label_key=label_key,
                                               connect_key = 'spatial_connectivities', 
                                               LOWER_BOUND=3, 
                                               center_weight=center_weight, 
                                               feature_range=feature_range)
    print("Done!")
    
    ## Manifold space KNNEnergy
    print("---Find K-Nearest Neighbor in UMAP:", end=' ')
    adata = _find_knn(adata)
    print("Done!")
    print("---Compute KNN Energy:", end=' ')
    KNNEnergy, _ = _local_energy(adata, 
                                 label_key=label_key,
                                 connect_key = 'knn_connectivities', 
                                 center_weight=center_weight, 
                                 LOWER_BOUND=0.05, 
                                 feature_range=feature_range)
    print("Done!")

    ## Co occurence GlobalEnergy
    print("---Compute Global Energy:", end=' ')
    GlobalEnergy, LabelPairCount = _global_energy(adata, 
                                                  RegionCount, 
                                                  label_key=label_key,
                                                  connect_key = 'spatial_connectivities',
                                                  LOWER_BOUND=1, 
                                                  feature_range=feature_range)
    print("Done!")
    return GlobalEnergy, KNNEnergy, SpatialEnergy