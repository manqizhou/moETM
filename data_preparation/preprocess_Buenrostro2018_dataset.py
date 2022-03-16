import pdb 
import os
import os.path as osp

import anndata as ad 
from scipy.sparse import csc_matrix, coo_matrix
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# conda install -c bioconda pyranges
# pip install pyranges
import pyranges as pr

import gzip
from scipy.io import mmread


data_path_prefix = "../data/Buenrostro2018"

'''
Preprocess ATAC data
'''

# read ATAC peak data 
filename = 'raw_data/GSE96769_PeakFile_20160207.bed.gz'
genomic_region_df = pd.read_table(osp.join(data_path_prefix, filename), header=None)
genomic_region_df = genomic_region_df.iloc[:,:4]
genomic_region_df.columns = ["chr", "start", "end", "strand"]

# add enumerated peak names as indices to the peak table
old_idx = genomic_region_df.index.tolist()
new_idx = [f"Peak {i}" for i in range(1, genomic_region_df.shape[0]+1)]
idx_mapping_dct = {old:new for old, new in zip(old_idx, new_idx)}
genomic_region_df.rename(index=idx_mapping_dct, inplace=True)

# read preprocessed atac sample names 
filename = "preprocessed.GSE96769_sample_names.tsv"
sample_name_df = pd.read_csv(osp.join(data_path_prefix, filename), sep="\t")

# read atac read counts
filename = 'raw_data/GSE96769_scATACseq_counts.txt.gz'
columns, rows, counts = np.loadtxt(osp.join(data_path_prefix, filename), unpack=True, dtype=int)
X = coo_matrix((counts, (rows-1, columns-1)), shape=(rows.max(), columns.max()))
adata = ad.AnnData(X=X.tocsc(), obs=sample_name_df.iloc[:,:5], var=genomic_region_df)
filename = "preprocessed.Buenrostro2018.atac_ad.h5ad"
adata.write(osp.join(data_path_prefix, filename))
