import os
import time
import squidpy as sq
import pandas as pd
import numpy as np
from pianno._calculate_energy import CalculateEnergy

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = (os.dup(1), os.dup(2))

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        os.close(self.null_fds[0])
        os.close(self.null_fds[1])

def PrioriInitialization(adata, label_key='InitialLabel', omegaK=0.99, omegaG=0.01, 
                         energy_scale=3, feature_range=1, center_weight=1,
                         coord_type=None, radius=None, n_rings=1, n_neighs=6, fig_save_path=None, visual=True):
    """Modeling the initial value of priori.

    Args:
        adata (Anndata): Pianno object.
        label_key (str, optional): Labels of adata used to build a Markov Random Field. Defaults to 'InitialLabel'.
        omegaK (float, optional): The larger the omegaK, the larger the effect of gene expression similarity on the initial value of priori. Defaults to 0.99.
        omegaG (float, optional): the larger the omegaG, the larger the effect of initial annotation. Defaults to 0.01.
        energy_scale (int, optional): A scale factor to adjust the strength of the priori. Defaults to 3.
        feature_range (int, optional): Desired range of transformed data during Min-Max normalization. Defaults to 1.
        center_weight (int, optional): Weighting of center spot label. Defaults to 1.
        coord_type (_type_, optional): Type of coordinate system. Defaults to None. Valid options are:
            - `{c.GRID.s!r}` - grid coordinates.
            - `{c.GENERIC.s!r}` - generic coordinates.
            - `None` - `{c.GRID.s!r}` if ``spatial_key`` is in :attr:`anndata.AnnData.uns`
              with ``n_neighs = 6`` (Visium), otherwise use `{c.GENERIC.s!r}`.
        radius (_type_, optional): Only available when ``coord_type = {c.GENERIC.s!r}``. Depending on the type:
            - :class:`float` - compute the graph based on neighborhood radius.
            - :class:`tuple` - prune the final graph to only contain edges in interval `[min(radius), max(radius)]`.
        n_rings (int, optional): Number of rings of neighbors for grid data. Only used when ``coord_type = {c.GRID.s!r}``. Defaults to 1.
        n_neighs (int, optional): number of neighborhoods. Defaults to 6.
        fig_save_path (_type_, optional): Save path of generated figures. Defaults to None.

    Returns:
        Anndata: Pianno object.
    """
    # start = time.time()
    # Calculate Enargy
    label_key=label_key
    n_rings=n_rings
    n_neighs=n_neighs
    coord_type=coord_type
    radius=radius
    center_weight=center_weight
    feature_range=feature_range
    GlobalEnergy, KNNEnergy, SpatialEnergy = CalculateEnergy(adata, label_key=label_key, 
                                                             n_rings=n_rings, n_neighs=n_neighs, 
                                                             coord_type=coord_type, radius=radius,
                                                             center_weight=center_weight, 
                                                             feature_range=feature_range)
    Energy = {"SpatialEnergy":SpatialEnergy, "KNNEnergy":KNNEnergy, "GlobalEnergy":GlobalEnergy,
              "Energy_key":{"omegaK": omegaK, "omegaG": omegaG, "energy_scale": energy_scale}}
    E1 = (1-omegaK)*SpatialEnergy + omegaK*KNNEnergy
    PairwiseEnergy = pd.DataFrame(((1-omegaG)*E1 + omegaG*GlobalEnergy)*energy_scale)
    Prior = np.exp(-PairwiseEnergy).div(np.exp(-PairwiseEnergy).sum(1),0)
    
    Regions = adata.uns['PatternList'].columns.to_list()
    title = [r+' Priori' for r in Regions]
    adata.obs[title] = np.array(Prior)
    
    Mask = adata.uns['Mask'].copy()
    PriorImage = []
    Prior = np.array(Prior)
    for i in range(len(Regions)):
        bg = np.zeros(Mask.shape).flatten()
        bg[adata.obs['spotID']] = Prior[:,i]
        img = bg.reshape(Mask.shape)*Mask
        PriorImage.append(img)
    adata.uns[label_key+'_PriorImage'] = PriorImage
    adata.uns[label_key+'_Energy'] = Energy
    
    if adata.uns['unknown_type'] == 'Background':
        title.remove('Background Priori')
    if visual:
        with suppress_stdout_stderr():
            sq.pl.spatial_scatter(adata, color=title, size=None, shape=None, ncols=len(title),
                                save=fig_save_path)
    del GlobalEnergy, KNNEnergy, SpatialEnergy
    # end = time.time()
    # time_sum = end-start
    # print('Elapsed time: %0.3fs'%time_sum)
    return adata