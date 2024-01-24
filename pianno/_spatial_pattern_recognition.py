'''
Recognize spatial pattern.
'''
import squidpy as sq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv
from skimage import morphology as sm
from skimage import exposure
from skimage import measure
from skimage import filters
from skimage.restoration import denoise_tv_chambolle
from skimage.filters import unsharp_mask

from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

## _create_pattern_list
def _create_pattern_list(Patterndict):
    unknown_type = []
    for k, v in Patterndict.items():
        if len(v) == 0:
            unknown_type.append(k)
    if len(unknown_type) > 1:
        raise KeyError('More than one domain without markers. Please recheck your Patterndict')
    elif len(unknown_type) == 0:
        unknown_type = None
    else:
        unknown_type = unknown_type[0]

    keys =  list(Patterndict.keys())
    Regions = [l.split('_')[0] for l in keys]
    Patterns = set([p for lis in Patterndict.values() for p in lis])
    PatternList = pd.DataFrame(0,index=Patterns, columns=Regions)
    for i in range(len(Regions)):
        PatternList.loc[Patterndict[keys[i]],Regions[i]] = 1
    PatternList = PatternList.sort_values(by=Regions, ascending=False)
    return PatternList, unknown_type

## _median_filtering
def _median_filtering(adata, Mask=None, Exp=None):
    gene = np.zeros(Mask.shape).flatten()
    gene[adata.obs['spotID']] = Exp
    img = gene.reshape(Mask.shape)*Mask
    img = filters.median(img,sm.disk(1))
    return img

## _pattern_image
def _pattern_image(adata, Patterndict=None, layer='DenoisedX'):
    # Create pattern image
    PatternList, unknown_type = _create_pattern_list(Patterndict)
    adata.uns['PatternList'] = PatternList.copy()
    adata.uns['unknown_type'] = unknown_type
    Regions = PatternList.columns.to_list()
    if unknown_type:
        Regions.remove(unknown_type)
        del Patterndict[unknown_type]
    # subset
    Patterns = PatternList.index.to_list()
    DenoisedX = adata[:,Patterns].layers[layer].copy()

    Mask = adata.uns['Mask']
    MedianGene = list(map(lambda x:_median_filtering(adata, Mask, x),DenoisedX.T))
    MedianGene = np.array(MedianGene)
    MedianPattern = [np.median(MedianGene[PatternList[r]==1],axis=0) for r in Regions]
    MedianPattern = np.array(MedianPattern)

    keys = [l.split('_') for l in Patterndict.keys()]
    index = [[Regions.index(r) for r in k] for k in keys]
    PI = []
    for idx in index:
        if len(idx) == 1:
            img = MedianPattern[idx[0]]
            #img = exposure.rescale_intensity(img)
            PI.append(img)
        else:
            img = MedianPattern[idx[0]]
            for i in idx[1:]:
                img = img - MedianPattern[i]
                #img = exposure.rescale_intensity(img)
            PI.append(img)
    adata.uns['PatternImage'] = PI
    return adata

## _pnregion_cluster
def _pnregion_cluster(label_image,weight_image,n_clusters=2):
    labels = np.unique(label_image)[np.unique(label_image)>0]
    ConnectedRegions = []
    Weights = []
    for l in labels:
        connectedregion = 1*(label_image == l)
        ConnectedRegions.append(connectedregion)
        weight = np.average(weight_image,weights=connectedregion)
        Weights.append(weight)
        
    Weights = np.array(Weights)
    Weights_idx = np.argsort(-Weights)
    WeightsX = Weights[Weights_idx]
    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(WeightsX.reshape(-1, 1))
    label_pred = estimator.labels_ 

    PWeights_idx = Weights_idx[label_pred == np.argmax(estimator.cluster_centers_)]
    NWeights_idx = Weights_idx[label_pred == np.argmin(estimator.cluster_centers_)]
    PConnectedRegions = [ConnectedRegions[CR] for CR in PWeights_idx]
    NConnectedRegions = [ConnectedRegions[CR] for CR in NWeights_idx]
    Pregion = np.sum(PConnectedRegions,axis = 0)
    Nregion = np.sum(NConnectedRegions,axis = 0)
    return Pregion,Nregion

## _feature_image
def _feature_image(adata, n_class=3, small_objects=2, dilation_radius=2, denoise_weight=0.2, 
                   unsharp_radius=5, unsharp_amount=10, gaussian_blur=5):
    PI = np.array(adata.uns['PatternImage'])
    FI = []
    Mask = np.array(adata.uns['Mask'])
    for i in range(PI.shape[0]):
        img = PI[i]
        thres = filters.threshold_multiotsu(img,classes=n_class).max()
        Pregion = img > thres
        Pregion = sm.remove_small_objects(Pregion,min_size=small_objects,connectivity=1)
        img = denoise_tv_chambolle(img*Pregion, weight=denoise_weight)

        # cluster
        Pregion = Pregion*Mask
        label_PPImask = measure.label(Pregion,connectivity=2)
        if label_PPImask.max() > 2:
            pregion = (1*Mask - 1*Pregion) > 0
            pregion = sm.binary_erosion(pregion,sm.square(12))
            label_PPImask = label_PPImask + pregion*int(label_PPImask.max()+1)
            Pregion,nregions = _pnregion_cluster(label_image=label_PPImask,weight_image=img,n_clusters=2)
        
        Pregion = sm.binary_dilation(Pregion, sm.disk(dilation_radius))
        unsharpPI = unsharp_mask(PI[i], radius=unsharp_radius, amount=unsharp_amount)
        gaussianResult1 = cv.GaussianBlur(unsharpPI,(3,3),gaussian_blur)
        img = denoise_tv_chambolle(gaussianResult1*Pregion, weight=denoise_weight)
        img = exposure.rescale_intensity(img)####
        FI.append(img)
    adata.uns['FeatureImage'] = FI
    return adata

## _label_assign
def _label_assign(adata, uncertainty=0.5, label_key='InitialLabel'):
    unknown_type = adata.uns['unknown_type']
    FI = adata.uns['FeatureImage'].copy()
    if unknown_type != None:
        Mask = adata.uns['Mask'].copy()
        bg = np.ones(Mask.shape)*uncertainty - np.array(FI).sum(0)
        FI.append(bg*Mask)
    Score = np.array([F.flatten()[adata.obs['spotID']] for F in FI]).T
    Regions = adata.uns['PatternList'].columns.to_list()
    InitialLabel = pd.Series(np.argmax(Score,1), dtype="category")
    InitialLabel.cat.categories = Regions[0:len(set(InitialLabel))]
    InitialLabel.index = adata.obs.index
    adata.obs[label_key] = InitialLabel
    adata.uns['FeatureImage'] = FI
    return adata

## _skeleton_extracting
def _skeleton_extracting(adata, label_key='InitialLabel'):
    Regions = adata.uns['PatternList'].columns.to_list()
    Skeleton = pd.Series([None]*adata.shape[0], dtype="category")
    Skeleton.index = adata.obs.index
    Skeleton = Skeleton.cat.set_categories(Regions)
    RegionOneHot = pd.get_dummies(adata.obs[label_key])
    Mask = adata.uns['Mask']
    for r in Regions:
        bg = np.zeros(Mask.shape).flatten()
        bg[adata.obs['spotID']] = RegionOneHot[r]
        img = bg.reshape(Mask.shape)*Mask
        img = sm.skeletonize(img)
        Skeleton[img.flatten()[adata.obs['spotID']]] = r
    adata.obs[label_key+'_Skeleton'] = Skeleton
    return adata

## mask_background
def _mask_background(adata, label_key='InitialLabel'):
    InitialLabel = adata.obs[label_key].copy()
    InitialLabel[InitialLabel == adata.uns['unknown_type']] = None
    InitialLabel = InitialLabel.cat.remove_categories('Background')
    adata.obs[label_key] = InitialLabel
    return adata

## SpatialPatternRecognition
def SpatialPatternRecognition(adata, Patterndict=None,
                              layer='DenoisedX',
                              n_class=3,
                              small_objects=2,
                              dilation_radius=2,
                              denoise_weight=0.2,
                              unsharp_radius=5, 
                              unsharp_amount=10, 
                              gaussian_blur=5,
                              label_key='InitialLabel',
                              uncertainty=0.5,
                              visual=True,
                              fig_width = 4,
                              fig_height = 8,
                              fontsize = 24,
                              fig_save_path=None):
    """A Pianno object is initialized based on the raw gene counts matrix and the coordinates of the spatial spots. pre-processing, including calculation of size factor and quality control, is also performed.

    Args:
        adata (Anndata): Pianno object.
        Patterndict (dict, optional): Marker list. Defaults to None.
        layer (str, optional): Layer of adata to use as denoised expression values. Defaults to 'DenoisedX'.
        n_class (int, optional): The number of levels of pixels to be segmented in the pattern image. Defaults to 3.
        small_objects (int, optional): Area threshold for small regions to filter out. Defaults to 2.
        dilation_radius (int, optional): Dilation radius. Defaults to 2.
        denoise_weight (float, optional): Denoising weight. The greater `denoise_weight`, the more denoising (at the expense of fidelity to `input`). Defaults to 0.2.
        unsharp_radius (int, optional): Radius for image sharpening. Note that 0 radius means no blurring, and negative values are not allowed. Defaults to 5.
        unsharp_amount (int, optional): Amount for image sharpening. The details will be amplified with this factor. The factor could be 0 or negative. Defaults to 10.
        gaussian_blur (int, optional): Gaussian kernel standard deviation for Gaussian Blur in OpenCV. Defaults to 5.
        label_key (str, optional): Key name of the generated label. Defaults to 'InitialLabel'.
        uncertainty (float, optional): Threshold of uncertainty. Defaults to 0.5.
        fig_width (int, optional): Figure width. Defaults to 4.
        fig_height (int, optional): Figure height. Defaults to 8.
        fontsize (int, optional): fontsize. Defaults to 24.
        fig_save_path (_type_, optional): Save path of generated figures. Defaults to None.

    Returns:
        Anndata: Pianno object.
    """

    Patterndict=Patterndict
    n_class=n_class
    small_objects=small_objects
    dilation_radius=dilation_radius
    denoise_weight=denoise_weight
    unsharp_radius=unsharp_radius
    unsharp_amount=unsharp_amount
    gaussian_blur=gaussian_blur
    label_key=label_key
    uncertainty=uncertainty
    layer=layer
    fontsize=fontsize

    adata = _pattern_image(adata, Patterndict=Patterndict, layer=layer)
    if n_class > 1:
        adata = _feature_image(adata, n_class=n_class, 
                               small_objects=small_objects, 
                               dilation_radius=dilation_radius, 
                               denoise_weight=denoise_weight,
                               unsharp_radius=unsharp_radius, 
                               unsharp_amount=unsharp_amount, 
                               gaussian_blur=gaussian_blur)
    elif n_class == 1:
        adata.uns['FeatureImage'] = adata.uns['PatternImage'].copy()
    adata = _label_assign(adata,uncertainty=uncertainty,
                          label_key=label_key)

    adata = _skeleton_extracting(adata, label_key=label_key)
    
    if adata.uns['unknown_type'] == 'Background':
        adata = _mask_background(adata, label_key=label_key)
    
    PI = adata.uns['PatternImage']
    FI = adata.uns['FeatureImage']
    NR = len(PI)
    Regions = adata.uns['PatternList'].columns.to_list()
    if visual:
        fig_width = 4
        fig_height = 6
        fig = plt.figure(figsize=(fig_width*(NR+1), fig_height))
        for i in range(NR):
            ax = plt.subplot2grid((2, NR+2), (0, i))
            plt.imshow(PI[i], cmap='RdBu_r')
            #sns.heatmap(PI[i], cmap=sns.diverging_palette(220, 20, n=200), cbar=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                ax.set_ylabel('Pattern', fontsize=fontsize, labelpad=5)
            ax.set_title(Regions[i],fontsize=fontsize)
            ax = plt.subplot2grid((2, NR+2), (1, i))
            plt.imshow(FI[i], cmap='RdBu_r')
            #sns.heatmap(FI[i], cmap=sns.diverging_palette(220, 20, n=200), cbar=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.xticks([])
            plt.yticks([])
            if i == 0:
                ax.set_ylabel('Feature', fontsize=fontsize, labelpad=5)
        ax = plt.subplot2grid((2, NR+2), (0, NR), colspan=2, rowspan=2)
        
        if label_key+'_colors' in adata.uns.keys(): 
            del adata.uns[label_key+'_colors']    
        sq.pl.spatial_scatter(adata, color=label_key, size=None, shape=None,
                            legend_fontsize=fontsize, ax=ax, legend_na=False)    
        ax.set_title(label_key,fontsize=fontsize)
        ax.set_xlabel('')
        ax.set_ylabel('')
        if fig_save_path != None:
            plt.savefig(fig_save_path)
        plt.show()
    if adata.uns['unknown_type'] == 'Background':
        adata.obs[label_key] = adata.obs[label_key].cat.add_categories(['Background'])
    return adata

## SpatialPatternRecognition for Hyperparameter Tuning
def spatial_pattern_recognition(adata, Patterndict=None,
                              layer='DenoisedX',
                              n_class=3,
                              small_objects=2,
                              dilation_radius=2,
                              denoise_weight=0.2,
                              unsharp_radius=5, 
                              unsharp_amount=10, 
                              gaussian_blur=5,
                              label_key='InitialLabel',
                              uncertainty=0.5):
    """A Pianno object is initialized based on the raw gene counts matrix and the coordinates of the spatial spots. pre-processing, including calculation of size factor and quality control, is also performed.

    Args:
        adata (Anndata): Pianno object.
        Patterndict (dict, optional): Marker list. Defaults to None.
        layer (str, optional): Layer of adata to use as denoised expression values. Defaults to 'DenoisedX'.
        n_class (int, optional): The number of levels of pixels to be segmented in the pattern image. Defaults to 3.
        small_objects (int, optional): Area threshold for small regions to filter out. Defaults to 2.
        dilation_radius (int, optional): Dilation radius. Defaults to 2.
        denoise_weight (float, optional): Denoising weight. The greater `denoise_weight`, the more denoising (at the expense of fidelity to `input`). Defaults to 0.2.
        unsharp_radius (int, optional): Radius for image sharpening. Note that 0 radius means no blurring, and negative values are not allowed. Defaults to 5.
        unsharp_amount (int, optional): Amount for image sharpening. The details will be amplified with this factor. The factor could be 0 or negative. Defaults to 10.
        gaussian_blur (int, optional): Gaussian kernel standard deviation for Gaussian Blur in OpenCV. Defaults to 5.
        label_key (str, optional): Key name of the generated label. Defaults to 'InitialLabel'.
        uncertainty (float, optional): Threshold of uncertainty. Defaults to 0.5.

    Returns:
        Anndata: Pianno object.
    """

    Patterndict=Patterndict
    n_class=n_class
    small_objects=small_objects
    dilation_radius=dilation_radius
    denoise_weight=denoise_weight
    unsharp_radius=unsharp_radius
    unsharp_amount=unsharp_amount
    gaussian_blur=gaussian_blur
    label_key=label_key
    uncertainty=uncertainty
    layer=layer

    adata = _pattern_image(adata, Patterndict=Patterndict, layer=layer)
    if n_class > 1:
        adata = _feature_image(adata, n_class=n_class, 
                               small_objects=small_objects, 
                               dilation_radius=dilation_radius, 
                               denoise_weight=denoise_weight,
                               unsharp_radius=unsharp_radius, 
                               unsharp_amount=unsharp_amount, 
                               gaussian_blur=gaussian_blur)
    elif n_class == 1:
        adata.uns['FeatureImage'] = adata.uns['PatternImage'].copy()
    adata = _label_assign(adata,uncertainty=uncertainty,
                          label_key=label_key)

    adata = _skeleton_extracting(adata, label_key=label_key)
    
    if adata.uns['unknown_type'] == 'Background':
        adata = _mask_background(adata, label_key=label_key)

    if label_key+'_colors' in adata.uns.keys(): 
        del adata.uns[label_key+'_colors']    
    if adata.uns['unknown_type'] == 'Background':
        adata.obs[label_key] = adata.obs[label_key].cat.add_categories(['Background'])
    return adata
