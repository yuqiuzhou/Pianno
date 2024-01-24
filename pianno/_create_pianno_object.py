'''
Create a Pianno object from raw count matrix and spatial coordinates.
'''
import scanpy as sc
import numpy as np
import pandas as pd
import warnings
from anndata import AnnData
warnings.filterwarnings("ignore")
import rpy2.robjects as robjects

## _initialize_adata
def _initialize_adata(count_matrix=None, coordinates=None, data_path=None, count_file=None, min_spots_prop=0.01):
    """create Anndata object, QC, filter_genes, filter_cells

    Args:
        count_matrix (array, optional): A spot-by-gene raw counts matrix. Defaults to None.
        coordinates (array, optional): A spatial coordinates dataframe. Defaults to None.
        data_path (str, optional): Path to the root directory containing Visium files. Defaults to None.
        count_file (str, optional): Which file in the passed directory to use as the count file. Defaults to None.
        min_spots_prop (float, optional): Minimum proportion of spots expressed required for a gene to pass filtering. Defaults to 1%.

    Returns:
        Anndata: _description_
    """
    count_matrix=count_matrix
    if type(coordinates) == pd.core.frame.DataFrame:
        coordinates=coordinates.values
    else:
        coordinates=coordinates
    data_path=data_path
    count_file=count_file
       
    if data_path != None and count_file != None:
        adata  = sc.read_visium(data_path,count_file = count_file)
    else:
        adata = AnnData(count_matrix, obsm={"spatial": coordinates})
    adata.var_names_make_unique()
    #adata.var["mt"] = adata.var_names.str.startswith("MT-")
    #adata = adata[:,~adata.var['mt']]
    sc.pp.filter_genes(adata, min_cells=int(adata.shape[0] * min_spots_prop)) 
    sc.pp.filter_cells(adata, min_counts=int(1)) 
    if type(adata.X) == np.ndarray:
        adata.layers['RawX'] = adata.X.copy()
    else:
        adata.layers['RawX'] = adata.X.A.copy()    
    return adata

## SizeFactor
def ComputeSizeFactor(adata):
    robjects.r('''
    ComputeSizeFactor <- function(RawMatrix){
      if(require(scran)){
        print("load scran successfully")
      }else{
        print("scran does not exist, trying to install......")
        if (!require("BiocManager", quietly = TRUE))
          install.packages("BiocManager")
        BiocManager::install("scran")
        if(require("scran")){
          print("successfully installed and loaded")
        } else {
          stop("fail to install scran")
        }
      }

      # row-gene col-spot
      RawMatrix = t(RawMatrix)
      sceset <- SingleCellExperiment(assays = list(counts = RawMatrix))
      qclust <- quickCluster(sceset)
      sceset <- computeSumFactors(sceset, clusters = qclust)
      s <- sizeFactors(sceset)

      return(s)
    }
    ''')
    
    RawX = adata.layers['RawX'].copy()
    rawmatrix = robjects.FloatVector(RawX.flatten())
    mat_input=robjects.r['matrix'](rawmatrix, ncol=RawX.shape[1], byrow=True)
    sizefactor=robjects.r['ComputeSizeFactor'](mat_input)   
    SizeFactor = np.array(sizefactor)
    adata.obs['SizeFactor'] = SizeFactor
    return adata

## CreatePiannoObject
def CreatePiannoObject(count_matrix=None, coordinates=None, 
                       data_path=None, count_file=None, 
                       min_spots_prop=0.01):
    """Create a Pianno object from raw count matrix and spatial coordinates.

    Args:
        count_matrix (DataFrame, optional): Raw counts matrix, rows-spots, cols-genes. Defaults to None.
        coordinates (array, optional): . (x, y) or (col, row).
        data_path (str, optional): Path to the root directory containing *Visium* files. Defaults to None.
        count_file (str, optional): Which file in the passed directory to use as the count file. Typically either *filtered_feature_bc_matrix.h5* or *raw_feature_bc_matrix.h5*. Defaults to None.
        min_spots_prop (float, optional): _description_. Defaults to 1%.

    Returns:
        Anndata: Pianno object
    """
    count_matrix=count_matrix
    coordinates=coordinates
    data_path=data_path
    count_file=count_file
    
    # Create adata
    adata = _initialize_adata(count_matrix=count_matrix, coordinates=coordinates, 
                              data_path=data_path, count_file=count_file, 
                              min_spots_prop=min_spots_prop)
    # Size factor
    adata = ComputeSizeFactor(adata)    
    return adata