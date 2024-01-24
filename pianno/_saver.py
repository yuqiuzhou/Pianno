'''
Denoised by SAVER.
'''
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import rpy2.robjects as robjects
from sklearn import preprocessing

## SAVER
def SAVER(adata, layer_key='DenoisedX', ncores=1):
    robjects.r('''
    CreateSaverMatrix <- function(RawMatrix, ncores=1){
      if(require(SAVER)){
        print("load SAVER successfully")
      }else{
        print("SAVER does not exist, trying to install......")
        install.packages("SAVER")
        if(require("SAVER")){
          print("successfully installed and loaded")
        } else {
          stop("fail to install SAVER")
        }
      }
      RawMatrix = t(RawMatrix)
      SaverMatrix <- saver(RawMatrix, estimates.only = TRUE, ncores = ncores)
      
      return(SaverMatrix)
    }
    ''')
    
    RawX = adata.layers['RawX'].copy()
    rawmatrix = robjects.FloatVector(RawX.flatten())
    mat_input=robjects.r['matrix'](rawmatrix, ncol=RawX.shape[1], byrow=True)
    savermatrix=robjects.r['CreateSaverMatrix'](mat_input,ncores)
    
    SaverX = np.array(savermatrix).T
    #Min-Max normalization
    minmax = preprocessing.MinMaxScaler(feature_range = (0,1))
    SaverX = np.round(minmax.fit_transform(SaverX),2)    
    
    adata.layers[layer_key] = SaverX
    return adata