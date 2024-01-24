import squidpy as sq
import numpy as np
import pandas as pd
import cv2 as cv
from skimage import morphology as sm
from skimage import exposure
from skimage import measure
from skimage import filters
from skimage.restoration import denoise_tv_chambolle
from pianno._spatial_pattern_recognition import _pnregion_cluster
from pianno._spatial_pattern_recognition import _mask_background

def LabelRenew(adata, method='imgbased', keep_region=None, 
               raw_key='InitialLabel', new_key='RenewedLabel', 
               small_objects=2, blur_deviation=5, denoise_weight=0.2, 
               pattern_img='Priori', prob_img="Posteriori", visual=True):
    """Fine-tuning the initial annotation based on posterior probability.

    Args:
        adata (Anndata): Pianno object.
        method (str, optional): 'imgbased' or 'argmax'. Defaults to 'imgbased'.
        keep_region (list, optional): Regions that keep the original labels. Defaults to None.
        raw_key (str, optional): Original label key of adata. Defaults to 'InitialLabel'.
        new_key (str, optional): Renewed label key of adata. Defaults to 'RenewedLabel'.
        small_objects (int, optional): Area threshold for small regions to filter out. Defaults to 2.
        blur_deviation (int, optional): Gaussian kernel standard deviation for Gaussian Blur in OpenCV. Defaults to 5.
        denoise_weight (float, optional): Denoising weight. The greater `denoise_weight`, the more denoising (at the expense of fidelity to `input`). Defaults to 0.2.
        pattern_img (str, optional): Which image used as pattern image. Defaults to 'Priori'.
        prob_img (str, optional): Which image used as probability image. Defaults to "Posteriori".

    Returns:
        _type_: _description_
    """    
    adata_copy = adata.copy()
    adata_copy.obs['Original_'+raw_key] = adata_copy.obs[raw_key].copy()    
    Regions = adata_copy.uns['PatternList'].columns.to_list()
    pattern_img_key = raw_key + '_' + pattern_img[0:-1] + 'Image'
    prob_img_key = raw_key + '_' + prob_img[0:-1] + 'Image'

    if method=='argmax':
        prob = np.array(adata_copy.obs[[r+' '+prob_img for r in Regions]])
        reLabel = np.argmax(prob,1)
        Newlabel = pd.Series(reLabel, dtype="category")
        RL = [Regions[i] for i in Newlabel.cat.categories.to_list()]

    if method=='imgbased':
        ProbImage = adata_copy.uns[prob_img_key].copy()
        WI = adata_copy.uns['FeatureImage'].copy()
        PI = adata_copy.uns[pattern_img_key].copy()
        Mask = adata_copy.uns['Mask']

        Pregions = []
        FI = []
        for i in range(len(Regions)):
            img = PI[i]
            thres = filters.threshold_multiotsu(img,classes=2)
            Pregion = img > thres
            Pregion = sm.remove_small_objects(Pregion,min_size=small_objects,connectivity=1)
            # cluster
            Pregion = Pregion*Mask
            label_PPImask = measure.label(Pregion,connectivity=2)
            if label_PPImask.max() > 2:
                pregion = (1*Mask - 1*Pregion) > 0
                pregion = sm.binary_erosion(pregion,sm.square(12))
                label_PPImask = label_PPImask + pregion*int(label_PPImask.max()+1)
                Pregion,nregions = _pnregion_cluster(label_image=label_PPImask,weight_image=WI[i],n_clusters=2)        
            Pregion = sm.binary_dilation(Pregion, sm.disk(1))
            Pregion = sm.binary_erosion(Pregion, sm.disk(2))                
            Pregions.append(Pregion)

            img = exposure.rescale_intensity(ProbImage[i])*Pregion
            img = filters.median(img,sm.square(2))
            img = cv.GaussianBlur(img,(3,3),blur_deviation)
            img = denoise_tv_chambolle(img, weight=denoise_weight)
            img = exposure.rescale_intensity(img)####      
            FI.append(img)        

        nrow, ncol = Mask.shape
        prob1 = np.array(FI).reshape((len(Regions), nrow*ncol)).T
        prob1 = prob1[adata_copy.obs['spotID'].values]
        Newlabel = pd.Series(np.argmax(prob1,1), dtype="category")
        RL = [Regions[i] for i in Newlabel.cat.categories.to_list()]
        
    Newlabel.index = adata_copy.obs.index    
    Newlabel.cat.categories = RL 
    
    if keep_region != None:
        for kr in keep_region:
            if kr not in RL:
                RL.append(kr)
                Newlabel = Newlabel.cat.add_categories(kr)
        RL = [k for k in Regions if k in RL]
        Newlabel = Newlabel.cat.reorder_categories(RL)
        InitialLabel = adata_copy.obs['Original_'+raw_key]
        keep_label = [idx for r in keep_region for idx in InitialLabel[InitialLabel == r].index]
        Newlabel.index = adata_copy.obs.index 
        Newlabel[keep_label] = InitialLabel[keep_label]
        
    adata_copy.obs[new_key] = Newlabel
        
    if 'unknown_type' in adata_copy.uns.keys(): 
        if adata_copy.uns['unknown_type'] == 'Background': 
            adata_copy = _mask_background(adata_copy, label_key=new_key)
            adata_copy = _mask_background(adata_copy, label_key='Original_'+raw_key)
    
    if 'Original_'+raw_key+'_colors' in adata_copy.uns.keys(): 
        del adata_copy.uns['Original_'+raw_key+'_colors']
    if new_key+'_colors' in adata_copy.uns.keys(): 
        del adata_copy.uns[new_key+'_colors']
    if visual:    
        sq.pl.spatial_scatter(adata_copy, color=['Original_'+raw_key, new_key], 
                              size=None, shape=None, title=[raw_key, 'ImprovedLabel'],
                              legend_na=False)    
    
    if 'unknown_type' in adata_copy.uns.keys(): 
        if adata_copy.uns['unknown_type'] == 'Background': 
            adata_copy.obs[new_key] = adata_copy.obs[new_key].cat.add_categories(['Background'])
            adata_copy.obs['Original_'+raw_key] = adata_copy.obs['Original_'+raw_key].cat.add_categories(['Background'])
    if new_key != raw_key:
        del adata_copy.obs['Original_'+raw_key]
    return adata_copy