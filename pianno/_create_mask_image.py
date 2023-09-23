'''
Create a mask image from the spatial coordinates.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

## _initialize_mask
def CreateMaskImage(adata, scale_factor=1):
    adata_copy = adata.copy()
    A = 'spatial' in adata_copy.uns.keys()
    B = 'array_col' in adata_copy.obs.keys()
    C = 'array_row' in adata_copy.obs.keys()
    if A and B and C:
        coordinates = adata_copy.obs[['array_row', 'array_col']]
    else:
        coordinates = pd.DataFrame(adata_copy.obsm['spatial'], index=adata_copy.obs.index,
                                   columns=['array_col', 'array_row'])    

    if (coordinates.max()>1000).any():
        coordinates = (coordinates/scale_factor).astype(int)

    evencol = coordinates[coordinates['array_col'] % 2 == 0]['array_row']
    oodcol = coordinates[coordinates['array_col'] % 2 != 0]['array_row']
    aligned = len(set(evencol).intersection(oodcol)) != 0
    coordinates = coordinates - coordinates.min()

    if aligned:
        nrow = int(coordinates['array_row'].max()) + 1 
        ncol = int(coordinates['array_col'].max()) + 1
        Mask = np.zeros((nrow, ncol))
        spotID = []
        for i in range(coordinates.shape[0]):   
            Mask[coordinates['array_row'][i], coordinates['array_col'][i]] = 1
            spotID.append(coordinates['array_row'][i]*ncol+coordinates['array_col'][i])
    else:
        nrow = coordinates['array_row'].max() + 1
        ncol = int((coordinates['array_col'].max() + 1)/2) + 1
        Mask = np.zeros((nrow, ncol))
        spotID = []
        for i in range(coordinates.shape[0]):   
            Mask[coordinates['array_row'][i], coordinates['array_col'][i] // 2] = 1
            spotID.append(coordinates['array_row'][i]*ncol+(coordinates['array_col'][i] // 2))
    
    adata_copy.uns['Mask'] = Mask
    coordinates['spotID'] = spotID
    coordinates = coordinates.loc[adata_copy.obs.index,:]
    adata_copy.obs[['array_col', 'array_row', 'spotID']] = coordinates
    plt.imshow(adata_copy.uns['Mask'], cmap="binary")
    plt.xticks([])
    plt.yticks([])
    plt.title('Mask Image',fontsize=20)
    plt.show()
    return adata_copy