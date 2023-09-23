import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tfw
tfw.compat.v1.disable_eager_execution()
tfw.get_logger().setLevel('ERROR')
import tensorflow_probability as tfp
import tensorflow.experimental.numpy as tnp
tnp.experimental_enable_numpy_behavior()
from random import sample
import time
import math
import squidpy as sq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc

def _avg_expression(adata, label_key='InitialLabel'): 
    Patterns = adata.uns['PatternList'].index.to_list()
    Tissue = adata[adata.obs[label_key+'_Skeleton']==adata.obs[label_key+'_Skeleton'],Patterns]
    InitialLabel = Tissue.obs[label_key].copy()
    X = Tissue.layers['RawX']
    Sn = Tissue.obs['SizeFactor'].values.reshape((-1,1))
    DF = pd.DataFrame(X/Sn,columns = Tissue.var.index, index = InitialLabel.index)
    del Tissue, X, Sn
    mu = DF.groupby(InitialLabel).mean()
    del DF, InitialLabel
    gc.collect()
    return mu

def _init_Bsg0(adata):
    Patterns = adata.uns['PatternList'].index.to_list()
    spatial_distances = adata.obsp['spatial_distances'].A
    Csg = adata[:,Patterns].layers['RawX'].copy()
    NS = Csg.shape[0]
    kernel_weight = (1/math.sqrt(2*math.pi))*(np.exp(-(spatial_distances**2)/2))*(np.identity(NS) + spatial_distances)
    Wss_init = np.round(kernel_weight/kernel_weight.sum(1).reshape(-1,1),2)
    Bsg0_init = np.matmul(Wss_init, Csg)
    del spatial_distances, Csg, kernel_weight, Wss_init
    gc.collect()
    return Bsg0_init

def entry_stop_gradients(target, mask):
    tf = tfw.compat.v1
    tf.disable_v2_behavior()
    mask_h = tf.logical_not(mask)
    mask = tf.cast(mask, dtype = target.dtype)
    mask_h = tf.cast(mask_h, dtype = target.dtype)
    return tf.add(tf.stop_gradient(tf.multiply(mask_h, target)), tf.multiply(mask, target))

def _mrf_inference(Pgr,
                   Csg,
                   Sn,
                   NS,
                   NG,
                   NR,
                   GlobalEnergy,
                   KNNEnergy,
                   SpatialEnergy,
                   omegaG_init = 0.01,
                   omegaK_init = 0.99,
                   min_omegaG = 0.01,
                   min_omegaK = 0.01,
                   energy_scale = 3,
                   Bgr = None,
                   Bsg0_init = None,
                   B = 10,
                   n_batches = 1,
                   rel_tol_adam = 1e-4,
                   rel_tol_em = 1e-4,
                   max_iter_adam = 1e5,
                   max_iter_em = 20,
                   learning_rate = 1e-3,
                   random_seed = None,
                   min_delta = 2,
                   dirichlet = 1e-2,
                   threads = 0,
                   shrinkage = True,
                   verbose = False,
                   gpu_id = None
                  ):
    if gpu_id != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    tf = tfw.compat.v1
    tf.disable_v2_behavior()
    tfd = tfp.distributions
    tf.reset_default_graph()

    # Data placeholders
    Csg_ = tf.placeholder(tf.float64, shape = (None, NG), name = "Csg_")
    Sn_ = tf.placeholder(tf.float64, shape = (None,), name = "Sn_")

    # Added for splines
    B = int(B)
    basis_means_fixed = np.linspace(np.min(Csg), np.max(Csg), B)
    basis_means = tf.constant(basis_means_fixed, dtype = tf.float64)
    b_init = 2 * (basis_means_fixed[1] - basis_means_fixed[0])**2

    LOWER_BOUND = 1e-10

    ## Spline variables
    a = tf.exp(tf.Variable(tf.zeros(shape = B, dtype = tf.float64)))
    b = tf.exp(tf.constant([-np.log(b_init)]*B, dtype = tf.float64))

    # Mean variables
    ## Shrinkage prior on delta
    if shrinkage:
        delta_log_mean = tf.Variable(0, dtype = tf.float64)
        delta_log_variance = tf.Variable(1, dtype = tf.float64) 

    ## Regular variables
    delta_log = tf.Variable(tf.random_uniform(shape = (NG,NR),
                                             minval = -2,
                                             maxval = 2,
                                             seed = random_seed,
                                             dtype = tf.float64),
                           dtype = tf.float64,
                           constraint = lambda x: tf.clip_by_value(x,
                                               tf.constant(np.log(min_delta),
                                                          dtype = tf.float64),
                                               tf.constant(np.infty, dtype = tf.float64))
                            )

    ## Prior Constant
    SpatialEnergy_ = tf.constant(SpatialEnergy, dtype = tf.float64)
    KNNEnergy_ = tf.constant(KNNEnergy, dtype = tf.float64)
    GlobalEnergy_ = tf.constant(GlobalEnergy, dtype = tf.float64)

    ## Prior variables
    omega_G = tf.Variable(omegaG_init, dtype = tf.float64, 
                          constraint = lambda x: tf.clip_by_value(x,min_omegaG,1.0))
    omega_K = tf.Variable(omegaK_init, dtype = tf.float64, 
                          constraint = lambda x: tf.clip_by_value(x,min_omegaK,1.0))
    
    E1 = omega_K*KNNEnergy_ + (1-omega_K)*SpatialEnergy
    Energy = (omega_G*GlobalEnergy_ + (1-omega_G)*E1)*energy_scale

    prior = tf.divide(tf.exp(-Energy),tf.expand_dims(tf.reduce_sum(tf.exp(-Energy),1), axis = 1))

    Pgr_ = tf.constant(Pgr, dtype = tf.float64)
    # Stop gradient for irrelevant entries of delta_log
    delta_log = entry_stop_gradients(delta_log, tf.cast(Pgr_, tf.bool))

    # Transformed variables
    delta = tf.exp(delta_log)
    prior_log = tf.log(prior)

    Sn__ = tf.tile(tf.expand_dims(tf.expand_dims(Sn_, axis = 1), axis = 2), (1, NG, NR))
    Bgr_ = tf.constant(Bgr, dtype = tf.float64)
    m1 = tf.multiply(delta, Bgr_) + LOWER_BOUND
    m2 = tf.constant(Bsg0_init, dtype = tf.float64)
    m1_ = tf.tile(tf.expand_dims(m1, axis = 0), [NS,1,1])
    m2_ = tf.tile(tf.expand_dims(m2, axis = 2), [1,1,NR])
    mu_sgr = tf.log(tf.add(m1_, m2_)) + tf.log(Sn__)

    mu_rsg = tf.transpose(mu_sgr, (2,0,1))
    mu_rsgb = tf.tile(tf.expand_dims(mu_rsg, axis = 3), (1, 1, 1, B))
    phi_rsg = tf.reduce_sum(a * tf.exp(-b * tf.square(mu_rsgb - basis_means)), 3) + LOWER_BOUND
    phi = tf.transpose(phi_rsg, (1,2,0))
    mu_sgr = tf.transpose(mu_rsg, (1,2,0))
    mu_sgr = tf.exp(mu_sgr)
    p = mu_sgr / (mu_sgr + phi)

    nb_pdf = tfd.NegativeBinomial(probs = p, total_count = phi)

    Csg_tensor_list = [Csg_ for r in range(NR)]
    Csg__ = tf.stack(Csg_tensor_list, axis = 2)

    c_log_prob_raw = nb_pdf.log_prob(Csg__)
    c_log_prob = tf.transpose(c_log_prob_raw, (0,2,1))
    c_log_prob_sum = tf.reduce_sum(c_log_prob, 2) + prior_log
    p_c_on_r_unorm = tf.transpose(c_log_prob_sum, (1,0))

    gamma_fixed = tf.placeholder(dtype = tf.float64, shape = (None,NR), name = "gamma_fixed")
    Q = -tf.einsum('nc,cn->', gamma_fixed, p_c_on_r_unorm)
    p_c_on_r_norm = tf.reshape(tf.reduce_logsumexp(p_c_on_r_unorm, 0), (1,-1))
    gamma = tf.transpose(tf.exp(p_c_on_r_unorm - p_c_on_r_norm))

    ## Priors
    if shrinkage:
        delta_log_prior = tfd.Normal(loc = delta_log_mean * Pgr_,
                                     scale = delta_log_variance)
        delta_log_prob = -tf.reduce_sum(delta_log_prior.log_prob(delta_log))

    THETA_LOWER_BOUND = 1e-20
    dirichlet_concentration = np.array([[dirichlet]*NR]*NS)
    prior_log_prior = tfd.Dirichlet(concentration = tf.constant(dirichlet_concentration,
                                                               dtype = tf.float64))
    prior_log_prob = -prior_log_prior.log_prob(tf.exp(prior_log) + THETA_LOWER_BOUND)

    ## End priors
    Q = Q + tf.reduce_sum(prior_log_prob)
    if shrinkage: Q = Q + delta_log_prob

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(Q)

    # Marginal log likelihood for monitoring convergence
    L_c = tf.reduce_sum(tf.reduce_logsumexp(p_c_on_r_unorm, 0))

    L_c = L_c - tf.reduce_sum(prior_log_prob)
    if shrinkage: L_c = L_c - delta_log_prob

    # Split the data
    sample_id = sample(range(NS), NS)
    sample_batch = [[sample_id[i] for i in range(NS) if i%n_batches == j] for j in range(n_batches)]
    splits = dict(zip(range(n_batches), sample_batch))

    # Start the graph and inference
    threads = int(threads)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads = threads,
                                 inter_op_parallelism_threads = threads)
    sess = tf.Session(config = session_conf)
    init = tf.global_variables_initializer()
    sess.run(init)

    fd_full = {Csg_: Csg, Sn_: Sn}
    log_liks = ll_old = sess.run(L_c, feed_dict = fd_full)
    log_liks_list = [log_liks]

    for i in range(max_iter_em):
        ll = 0 # log likelihood for this "epoch"
        for b in range(n_batches):
            fd = {Csg_: Csg[splits[0],], Sn_: Sn[splits[0]]}
            g = sess.run(gamma, feed_dict = fd)

            # M-step
            gfd = {Csg_: Csg[splits[0],], Sn_: Sn[splits[0]], gamma_fixed: g}

            Q_old = sess.run(Q, feed_dict = gfd)
            Q_diff = rel_tol_adam + 1
            mi = 0

            while mi < max_iter_adam and Q_diff > rel_tol_adam:
                mi = mi + 1
                sess.run(train, feed_dict = gfd)
                if mi % 20 == 0:
                    if verbose:
                        print([mi, sess.run(Q, feed_dict = gfd)])
                    Q_new = sess.run(Q, feed_dict = gfd)
                    Q_diff = -(Q_new - Q_old) / abs(Q_old)
                    Q_old = Q_new

            l_new = sess.run(L_c, feed_dict = gfd) # Log likelihood for this "epoch"
            ll = ll + l_new

        ll_diff = (ll - ll_old) / abs(ll_old)
        if verbose:
            print("%d old: %.3f; L new: %.3f; Difference: %.3f"%(mi, ll_old, ll, ll_diff))
        ll_old = ll
        log_liks_list.append(ll)

        if abs(ll_diff) < rel_tol_em:
            break

    print("log-likelihood: {}".format(log_liks_list[-1]))
    # Finished EM - peel off final values
    post_prob = sess.run(gamma, feed_dict = fd_full)
    sess.close()
    gc.collect()
    return post_prob

def PosterioriInference(adata,
                        label_key='InitialLabel',
                        min_omegaG = 0.01,
                        min_omegaK = 0.01,
                        min_delta = 2,
                        dirichlet = 1e-2,
                        learning_rate = 1e-3,
                        max_iter_adam = 1e5,
                        max_iter_em = 20,
                        B = 10,
                        n_batches = 1,
                        rel_tol_adam = 1e-4,
                        rel_tol_em = 1e-4,
                        random_seed = None,
                        threads = 0,
                        shrinkage = True,
                        verbose = False,
                        gpu_id = None,
                        fig_save_path = None
                        ):
    start = time.time()
    #check
    ## RawX
    if 'RawX' not in adata.layers.keys():
        raise KeyError('raw count matrix is not found in layers: RawX')    
    ## Size factor
    if 'SizeFactor' not in adata.obs.keys():
        raise KeyError('size factor vector is not found in obs: SizeFactor')
    ## PatternList
    Pgr = adata.uns['PatternList'].copy()
    Regions = Pgr.columns.to_list()
    Patterns = Pgr.index.to_list()
    
    ## Raw count matrix
    Csg = adata[:,Patterns].layers['RawX'].copy()
    
    ## Size factor
    Sn = np.array(adata.obs['SizeFactor'].values)
    
    ## Parameter
    NS = Csg.shape[0]
    NG = Csg.shape[1]
    NR = Pgr.shape[1]
    
    # Enargy
    Enargy = adata.uns[label_key+'_Energy']
    GlobalEnergy = Enargy["GlobalEnergy"]
    KNNEnergy = Enargy["KNNEnergy"]
    SpatialEnergy = Enargy["SpatialEnergy"]
    
    ## Bgr
    ske_mu_gr = _avg_expression(adata, label_key=label_key)
    Bgr = np.multiply(ske_mu_gr.T, Pgr)
    
    ## Bsg0_init
    Bsg0_init = _init_Bsg0(adata)
    
    # Inference
    omegaK_init = Enargy["Energy_key"]["omegaK"]
    omegaG_init = Enargy["Energy_key"]["omegaG"]
    energy_scale = Enargy["Energy_key"]["energy_scale"]
    min_omegaG = min_omegaG
    min_omegaK = min_omegaK
    min_delta = min_delta
    dirichlet = dirichlet
    learning_rate = learning_rate
    max_iter_adam = max_iter_adam
    max_iter_em = max_iter_em
    B = B
    n_batches = n_batches
    rel_tol_adam = rel_tol_adam
    rel_tol_em = rel_tol_em
    random_seed = random_seed
    threads = threads
    shrinkage = shrinkage
    verbose = verbose
    gpu_id = gpu_id
    fig_save_path = fig_save_path
    
    Posterior = _mrf_inference(Pgr = Pgr,
                               Csg = Csg,
                               Sn = Sn,
                               NS = NS,
                               NG = NG,
                               NR = NR,
                               GlobalEnergy = GlobalEnergy,
                               KNNEnergy = KNNEnergy,
                               SpatialEnergy = SpatialEnergy,
                               omegaG_init = omegaG_init,
                               omegaK_init = omegaK_init,
                               min_omegaG = min_omegaG,
                               min_omegaK = min_omegaK,
                               energy_scale = energy_scale,
                               Bgr = Bgr,
                               Bsg0_init = Bsg0_init,
                               B = B,
                               n_batches = n_batches,
                               rel_tol_adam = rel_tol_adam,
                               rel_tol_em = rel_tol_em,
                               max_iter_adam = max_iter_adam,
                               max_iter_em = max_iter_em,
                               learning_rate = learning_rate,
                               random_seed = random_seed,
                               min_delta = min_delta,
                               dirichlet = dirichlet,
                               threads = threads,
                               shrinkage = shrinkage,
                               verbose = verbose,
                               gpu_id = gpu_id
                              )   
    del Pgr, Csg, Sn, GlobalEnergy, KNNEnergy, SpatialEnergy, Bgr, Bsg0_init
    gc.collect()
    Regions = adata.uns['PatternList'].columns.to_list()
    title = [r+' Posteriori' for r in Regions]
    adata.obs[title] = np.array(Posterior)
    
    Mask = adata.uns['Mask'].copy()
    PosteriorImage = []
    Posterior = np.array(Posterior)
    for i in range(len(Regions)):
        bg = np.zeros(Mask.shape).flatten()
        bg[adata.obs['spotID']] = Posterior[:,i]
        img = bg.reshape(Mask.shape)*Mask
        PosteriorImage.append(img)
    adata.uns[label_key+'_PosteriorImage'] = PosteriorImage
    
    if 'unknown_type' in adata.uns.keys():
        if adata.uns['unknown_type'] == 'Background':
            title.remove('Background Posteriori')
    sq.pl.spatial_scatter(adata, color=title, size=None, shape=None, ncols=len(title),
                        save=fig_save_path)
    
    end = time.time()
    time_sum = end-start    
    print('Elapsed time: %0.3fs'%time_sum)
    plt.show()
    return adata