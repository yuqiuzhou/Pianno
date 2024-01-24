from pianno._posteriori_inference import PosterioriInference
from pianno._priori_initialization import PrioriInitialization
from pianno._label_renew import LabelRenew

def AnnotationImprovement(adata, 
                          raw_key='InitialLabel', 
                          new_key='RenewedLabel',
                          omegaK=0.99, omegaG=0.01,
                          method='imgbased',
                          keep_region=None,
                          blur_deviation=1, 
                          denoise_weight=0.01,
                          small_objects=2,
                          visual=True,
                          ):
    adata = PrioriInitialization(adata,
                                 label_key=raw_key,
                                 omegaK=omegaK, omegaG=omegaG,
                                 visual= visual)
    adata = PosterioriInference(adata, 
                                label_key=raw_key,
                                visual= visual)
    adata = LabelRenew(adata, method=method, 
                       keep_region=keep_region,
                       small_objects=small_objects,
                       blur_deviation=blur_deviation, 
                       denoise_weight=denoise_weight,
                       raw_key=raw_key, new_key=new_key,
                       visual= visual)
    return adata