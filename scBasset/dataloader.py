import os 
import pdb 
import pickle
import random
import numpy as np 
import pandas as pd

import anndata as ad
import scanpy as sc
from scipy.sparse import csc_matrix

import torch

import os.path as osp

from genome_utils import load_genome_to_dict

# TODO Update NIPS and mouse skin data loader functions to incorporate genome data


'''
Load and Prepare NIPS Challenge Dataset
'''

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


def prepare_nips_dataset(adata_gex, adata_mod2, batch_col="batch"):

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

'''
Load and Prepare Mouse skin dataset
'''

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

'''
Load and Prepare mouse brain dataset
'''

def load_mouse_brain_dataset(mod1_file_path, mod2_file_path, genome_dir=None, seqlen=1344, verbose=False):
    # load modalities
    adata_mod1 = ad.read_h5ad(mod1_file_path)
    adata_mod2 = ad.read_h5ad(mod2_file_path)

    # preprocess RNA count data and retain highly variable genes only
    sc.pp.normalize_total(adata_mod1, target_sum=1e4)
    sc.pp.log1p(adata_mod1)
    sc.pp.highly_variable_genes(adata_mod1)
    index = adata_mod1.var['highly_variable'].values

    adata_mod1 = ad.read_h5ad(mod1_file_path)
    adata_mod1 = adata_mod1[:, index]
    obs = adata_mod1.obs
    adata_mod1 = ad.AnnData(X=adata_mod1.X, obs=obs)

    # preprocess ATAC count data 
    sc.pp.normalize_total(adata_mod2, target_sum=1e4)
    sc.pp.log1p(adata_mod2)
    sc.pp.highly_variable_genes(adata_mod2)
    index = adata_mod2.var['highly_variable'].values

    adata_mod2 = ad.read_h5ad(mod2_file_path)
    adata_mod2 = adata_mod2[:, index]

    # if genome sequence data is requested
    if genome_dir is not None:

        # mouse: chr1-19 + chrX + chrY
        chr_filenames = [f"chr{id}.fa" for id in list(range(1,20)) + ["X", "Y"]]

        if verbose:
            print("> Loading genome...")

        # load mm10
        genome = load_genome_to_dict(genome_dir, chr_filenames)

        # get peaks
        peaks = adata_mod2.var.copy()

        # calculate length of peaks (all peaks are 500 bp)
        peaks["length"] = peaks["end"] - peaks["start"] + 1

        if verbose:
            print("> Extracting sequences underlying peaks...")

        # extend peaks 30bp both upstream downstream
        # during training we will stochastically shift sequences by 
        # stochastically removing 30bp from sequence start, end or both
        data_augmentation_offset = 30
        # peaks for this experiment are 500bp but we want 1344bp sequences
        # for scBasset training. we expand peak regions by (1344-500)//2
        # at each side
        requested_seqlen_offset = (seqlen - peaks["length"][0])//2
        peaks["sequence"] = [genome[chrom][start-data_augmentation_offset-requested_seqlen_offset:\
            end+data_augmentation_offset+requested_seqlen_offset+1]\
            for chrom, start, end in zip(peaks["chr"], peaks["start"], peaks["end"])]
        temp = [len(seq) for seq in peaks["sequence"]]
        assert np.mean(temp) == seqlen + 2 * data_augmentation_offset and np.std(temp) == 0

        # update length field
        peaks["length"] = seqlen + 2 * data_augmentation_offset
        
        var = peaks

    else:

        var_name = list(adata_mod2.var.index)
        var = pd.DataFrame(index=var_name)

    X = adata_mod2.X
    #X = X[:, np.array(np.argsort(X.sum(0))[::-1][0,0:20000]).squeeze()]
    obs = adata_mod2.obs

    adata_mod2 = ad.AnnData(X=X, obs=obs, var=var)
    
    return adata_mod1, adata_mod2

def prepare_mouse_brain_dataset(adata_gex, adata_mod2):

    batch_index = np.zeros(adata_gex.shape[0])

    obs = adata_gex.obs
    obs.insert(obs.shape[1], 'batch_indices', batch_index)
    adata_gex = ad.AnnData(X=adata_gex.X, obs=obs)

    obs = adata_mod2.obs
    obs.insert(obs.shape[1], 'batch_indices', batch_index)

    X = adata_mod2.X
    adata_mod2 = ad.AnnData(X=csc_matrix(X), obs=obs, var=adata_mod2.var)

    Index = np.array(X.sum(1) > 0).squeeze()

    adata_gex = adata_gex[Index]
    obs = adata_gex.obs
    adata_gex = ad.AnnData(X=adata_gex.X, obs=obs)

    adata_mod2 = adata_mod2[Index]
    obs = adata_mod2.obs
    adata_mod2 = ad.AnnData(X=adata_mod2.X, obs=obs, var=adata_mod2.var)

    return adata_gex, adata_mod2


'''
Load and prepare Buenrostro2018 scATAC-seq dataset
'''
def load_Buenrostro2018_dataset(atac_file_path, genome_dir=None, seqlen=1344, verbose=False):
    # load AnnData
    adata = ad.read_h5ad(atac_file_path)

    '''
    Remove some cells based on their batch indices and label

    original cell types:
    temp = ['HSC', 'MPP', 'LMPP', 'CMP', 'CLP', 'GMP', 'MEP',
    'MCP', 'mono', 'UNK', 'GMP3high', 'GMP2mid', 'GMP1low', 'pDC']

    Steps: 
        1 - Remove samples that do not have the word "singles" in their filename
        2 - Map samples having PB1022 and NaN to a group called other
        3 - Map GMP3high, GMP2mid and GMP   low to GMP cell type
        4 - Remove MCP cell type
    '''
    new_obs = adata.obs
    new_obs = new_obs.iloc[:,1:]

    # step 1
    new_obs = new_obs[new_obs.iloc[:,0].notnull()]
    new_obs = new_obs.iloc[:,1:]

    # step 2
    temp = new_obs.iloc[:,0].tolist()
    new_obs.iloc[:,0] =\
        [str(x).replace("nan", "other").replace("PB1022", "other")\
            for x in temp]

    # step 3
    new_obs.iloc[:,1] = new_obs.iloc[:,1].replace({x:"GMP" for x in ['GMP3high', 'GMP2mid', 'GMP1low']})
    
    # step 4
    # new_obs = new_obs[~new_obs.iloc[:,1].isin(["MCP"])]

    adata = adata[new_obs.index,:]
    adata.obs = new_obs

    # remove cells that do not have zero count for all peaks 
    # TODO may remove this
    sc.pp.filter_cells(adata, min_genes=100)

    # remove peaks that do not have zero count in all cell
    sc.pp.filter_genes(adata, min_counts=1)

    # remove peaks that are accessible in less than 1% of cells
    num_cells = adata.shape[0]
    sc.pp.filter_genes(adata, min_counts = (0.01 * num_cells) // 1)

    # preprocess ATAC count data 
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # if genome sequence data is requested
    if genome_dir is not None:

        # mouse: chr1-19 + chrX + chrY
        chr_filenames = [f"chr{id}.fa" for id in list(range(1,23)) + ["X", "Y"]]

        if verbose:
            print("> Loading genome...")

        # load hg19
        genome = load_genome_to_dict(genome_dir, chr_filenames)

        # get peaks
        peaks = adata.var.copy()

        # calculate length of peaks (all peaks are 500 bp)
        peaks["length"] = peaks["end"] - peaks["start"] + 1
        assert peaks["length"].unique().shape[0] == 1

        if verbose:
            print("> Extracting sequences underlying peaks...")

        # extend peaks 30bp both upstream downstream
        # during training we will stochastically shift sequences by 
        # stochastically removing 30bp from sequence start, end or both
        data_augmentation_offset = 30
        # peaks for this experiment are 500bp but we want 1344bp sequences
        # for scBasset training. we expand peak regions by (1344-501)//2 
        # at left side and (1344-501)//2 + 1 at right side
        requested_seqlen_offset = (seqlen - peaks["length"][0])//2
        if peaks["length"][0] % 2 == 1:
            odd_peak_length_offset = 1
        else:
            odd_peak_length_offset = 0
        peaks["sequence"] = [genome[chrom][start-data_augmentation_offset-requested_seqlen_offset:\
            end+data_augmentation_offset+requested_seqlen_offset+odd_peak_length_offset+1]\
            for chrom, start, end in zip(peaks["chr"], peaks["start"], peaks["end"])]
        temp = [len(seq) for seq in peaks["sequence"]]
        assert np.mean(temp) == seqlen + 2 * data_augmentation_offset and np.std(temp) == 0

        # update length field
        peaks["length"] = seqlen + 2 * data_augmentation_offset
        
        adata.var = peaks

    return adata

def prepare_Buenrostro2018_dataset(adata_atac):

    # rename columns of adata_atac.obs
    adata_atac.obs.columns =\
        ["batch_name", "cell_type", "experiment_medium", "n_genes"]

    # add cell types and batch_indices columns
    batch_name2idx = {x:idx for idx, x in\
        enumerate(['BM0106', 'BM0828', 'BM1077', 'BM1137', 'BM1214', 'other'])}
    batch_name2idx
    adata_atac.obs["batch_indices"] = [batch_name2idx[x] for x in\
        adata_atac.obs["batch_name"]]

    return adata_atac


if __name__ == "__main__":

    # # load and prepare nips dataset
    # nips_dataset_file_path = "../data/nips/newest/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"
    # adata_mod1, adata_mod2 = load_nips_dataset(nips_dataset_file_path) 
    # adata_mod1, adata_mod2 = prepare_nips_dataset(adata_mod1, adata_mod2)

    # # load and prepare Mouse skin dataset
    # mouse_skin_dataset_RNA_file_path = "../data/mouse_skin/rna_count_new.h5ad"
    # mouse_skin_dataset_ATAC_file_path = "../data/mouse_skin/atac_count_new.h5ad"
    # adata_mod1, adata_mod2 = load_mouse_skin_dataset(mouse_skin_dataset_RNA_file_path, mouse_skin_dataset_ATAC_file_path) 
    # adata_mod1, adata_mod2 = prepare_mouse_skin_dataset(adata_mod1, adata_mod2)

    # # load and prepare mouse brain dataset
    # mod1_file_path = f"../data/mouse_brain/rna_count.preprocessed.h5ad"
    # mod2_file_path = f"../data/mouse_brain/atac_count.preprocessed.h5ad"
    # genome_dir = "../data/mm10"
    # adata_mod1, adata_mod2 = load_mouse_brain_dataset(mod1_file_path, mod2_file_path,
    #     genome_dir=genome_dir, verbose=True) 
    # adata_mod1, adata_mod2 = prepare_mouse_brain_dataset(adata_mod1, adata_mod2)

    # load Buenrostro2018 dataset
    atac_data_file_path = f"../data/Buenrostro2018/preprocessed.Buenrostro2018.atac_ad.h5ad"
    genome_dir = "../data/hg19"
    adata_atac = load_Buenrostro2018_dataset(atac_data_file_path, genome_dir=genome_dir, verbose=True)
    adata_atac = prepare_Buenrostro2018_dataset(adata_atac)

    pdb.set_trace()
