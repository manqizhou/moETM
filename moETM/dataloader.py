
import numpy as np
import torch
import anndata as ad
import scanpy as sc
from scipy.sparse import csc_matrix

def load_mouse_skin_dataset(mod1_file_path, mod2_file_path):
    adata_mod1 = ad.read_h5ad(mod1_file_path)
    adata_mod2 = ad.read_h5ad(mod2_file_path)

    sc.pp.normalize_total(adata_mod1, target_sum=1e4)
    sc.pp.log1p(adata_mod1)
    sc.pp.highly_variable_genes(adata_mod1)
    index = adata_mod1.var['highly_variable'].values

    adata_mod1 = ad.read_h5ad(mod1_file_path)
    adata_mod1 = adata_mod1[:, index]
    obs = adata_mod1.obs
    adata_mod1 = ad.AnnData(X=adata_mod1.X, obs=obs)

    sc.pp.normalize_total(adata_mod2, target_sum=1e4)
    sc.pp.log1p(adata_mod2)
    sc.pp.highly_variable_genes(adata_mod2)
    index = adata_mod2.var['highly_variable'].values

    adata_mod2 = ad.read_h5ad(mod2_file_path)
    adata_mod2 = adata_mod2[:, index]
    X = adata_mod2.X
    #X = X[:, np.array(np.argsort(X.sum(0))[::-1][0,0:20000]).squeeze()]
    obs = adata_mod2.obs

    adata_mod2 = ad.AnnData(X=X, obs=obs)

    return adata_mod1, adata_mod2

def load_nips_dataset(mod_file_path):
    adata = ad.read_h5ad(mod_file_path)

    feature_gex_index = np.array(adata.var.feature_types) == 'GEX'
    feature_atac_index = np.array(adata.var.feature_types) == 'ATAC'

    obs = adata.obs
    adata_mod1 = ad.AnnData(X=adata.layers['counts'][:, feature_gex_index], obs=obs)
    adata_mod2 = ad.AnnData(X=adata.layers['counts'][:, feature_atac_index], obs=obs)

    adata_mod1_original = ad.AnnData.copy(adata_mod1)
    adata_mod2_original = ad.AnnData.copy(adata_mod2)

    sc.pp.normalize_total(adata_mod1, target_sum=1e4)
    sc.pp.log1p(adata_mod1)
    sc.pp.highly_variable_genes(adata_mod1)
    index = adata_mod1.var['highly_variable'].values

    adata_mod1 = ad.AnnData.copy(adata_mod1_original)
    adata_mod1 = adata_mod1[:, index]
    obs = adata_mod1.obs
    adata_mod1 = ad.AnnData(X=adata_mod1.X, obs=obs)

    sc.pp.normalize_total(adata_mod2, target_sum=1e4)
    sc.pp.log1p(adata_mod2)
    sc.pp.highly_variable_genes(adata_mod2)
    index = adata_mod2.var['highly_variable'].values

    adata_mod2 = ad.AnnData.copy(adata_mod2_original)
    adata_mod2 = adata_mod2[:, index]
    obs = adata_mod2.obs
    adata_mod2 = ad.AnnData(X=adata_mod2.X, obs=obs)

    return adata_mod1, adata_mod2

def prepare_nips_dataset(adata_gex, adata_mod2, batch_col = 'batch'):

    batch_index = np.array(adata_gex.obs[batch_col].values)
    unique_batch = list(np.unique(batch_index))
    batch_index = np.array([unique_batch.index(xs) for xs in batch_index])

    obs = adata_gex.obs
    obs.insert(obs.shape[1], 'batch_indices', batch_index)
    adata_gex = ad.AnnData(X=adata_gex.X, obs=obs)

    obs = adata_mod2.obs
    obs.insert(obs.shape[1], 'batch_indices', batch_index)

    X = adata_mod2.X
    #X = X[:, np.array(np.argsort(X.sum(0))[::-1][0,0:10000]).squeeze()]
    adata_mod2 = ad.AnnData(X=X, obs=obs)

    Index = np.array(X.sum(1)>0).squeeze()

    adata_gex = adata_gex[Index]
    obs = adata_gex.obs
    adata_gex = ad.AnnData(X=adata_gex.X, obs=obs)

    adata_mod2 = adata_mod2[Index]
    obs = adata_mod2.obs
    adata_mod2 = ad.AnnData(X=adata_mod2.X, obs=obs)

    return adata_gex, adata_mod2

def prepare_mouse_skin_dataset(adata_gex, adata_mod2):

    batch_index = np.zeros(adata_gex.shape[0])

    obs = adata_gex.obs
    obs.insert(obs.shape[1], 'batch_indices', batch_index)
    adata_gex = ad.AnnData(X=adata_gex.X, obs=obs)

    obs = adata_mod2.obs
    obs.insert(obs.shape[1], 'batch_indices', batch_index)

    X = adata_mod2.X
    adata_mod2 = ad.AnnData(X=csc_matrix(X), obs=obs)

    Index = np.array(X.sum(1) > 0).squeeze()

    adata_gex = adata_gex[Index]
    obs = adata_gex.obs
    adata_gex = ad.AnnData(X=adata_gex.X, obs=obs)

    adata_mod2 = adata_mod2[Index]
    obs = adata_mod2.obs
    adata_mod2 = ad.AnnData(X=adata_mod2.X, obs=obs)

    return adata_gex, adata_mod2

def data_process_moETM(adata_mod1, adata_mod2, n_sample, test_ratio=0.1):
    from sklearn.utils import resample
    Index = np.arange(0, n_sample)
    train_index = resample(Index, n_samples=int(n_sample*(1-test_ratio)), replace=False)
    test_index = np.array(list(set(range(n_sample)).difference(train_index)))

    train_adata_mod1 = adata_mod1[train_index]
    obs = train_adata_mod1.obs
    X = train_adata_mod1.X
    train_adata_mod1 = ad.AnnData(X=X, obs=obs)

    train_adata_mod2 = adata_mod2[train_index]
    obs = train_adata_mod2.obs
    X = train_adata_mod2.X
    train_adata_mod2 = ad.AnnData(X=X, obs=obs)

    test_adata_mod1 = adata_mod1[test_index]
    obs = test_adata_mod1.obs
    X = test_adata_mod1.X
    test_adata_mod1 = ad.AnnData(X=X, obs=obs)

    test_adata_mod2 = adata_mod2[test_index]
    obs = test_adata_mod2.obs
    X = test_adata_mod2.X
    test_adata_mod2 = ad.AnnData(X=X, obs=obs)

    ########################################################
    # Training dataset
    X_mod1 = np.array(train_adata_mod1.X.todense())
    X_mod2 = np.array(train_adata_mod2.X.todense())
    batch_index = np.array(train_adata_mod1.obs['batch_indices'])

    X_mod1 = X_mod1 / X_mod1.sum(1)[:, np.newaxis]
    X_mod2 = X_mod2 / X_mod2.sum(1)[:, np.newaxis]

    X_mod1_train_T = torch.from_numpy(X_mod1).float()
    X_mod2_train_T = torch.from_numpy(X_mod2).float()
    batch_index_train_T = torch.from_numpy(batch_index).to(torch.int64).cuda()

    # Testing dataset
    X_mod1 = np.array(test_adata_mod1.X.todense())
    X_mod2 = np.array(test_adata_mod2.X.todense())
    batch_index = np.array(test_adata_mod1.obs['batch_indices'])

    X_mod1 = X_mod1 / X_mod1.sum(1)[:, np.newaxis]
    X_mod2 = X_mod2 / X_mod2.sum(1)[:, np.newaxis]

    X_mod1_test_T = torch.from_numpy(X_mod1).float()
    X_mod2_test_T = torch.from_numpy(X_mod2).float()
    batch_index_test_T = torch.from_numpy(batch_index).to(torch.int64)

    del X_mod1, X_mod2, batch_index

    return X_mod1_train_T, X_mod2_train_T, batch_index_train_T, X_mod1_test_T, X_mod2_test_T, batch_index_test_T, test_adata_mod1






