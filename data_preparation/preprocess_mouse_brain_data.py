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


'''
NOTE: First download raw_data folder from moETM/Dataset/mouse_brain 
folder on OneDrive to data/mouse_brain folder in this repository
1
'''

data_path_prefix = "../data/mouse_brain"

'''
Preprocess ATAC data
'''
# read cell sample barcodes
filename = "raw_data/GSM4156599_brain.barcodes.txt"
f = open(osp.join(data_path_prefix, filename))
obs_bc = [line.strip() for line in f.readlines()]
f.close()

# read ATAC peak data 
filename = 'raw_data/GSM4156599_brain.peaks.bed.gz'
genomic_region_df = pr.read_bed(osp.join(data_path_prefix, filename)).df
genomic_region_df.rename(columns={
    "Chromosome": "chr",
    "Start": "start",
    "End": "end"
}, inplace=True)

# # find the minimum and maximum lengths of peaks provided in .bed file 
# genomic_region_df["peak_length"] =\
#     genomic_region_df["end"] - genomic_region_df["start"] + 1


# add enumerated peak names as indices to the peak table
old_idx = genomic_region_df.index.tolist()
new_idx = [f"Peak {i}" for i in range(1, genomic_region_df.shape[0]+1)]
idx_mapping_dct = {old:new for old, new in zip(old_idx, new_idx)}
genomic_region_df.rename(index=idx_mapping_dct, inplace=True)

# read ATAC peak read counts
filename = 'raw_data/GSM4156599_brain.counts.txt.gz'
X = csc_matrix(mmread(osp.join(data_path_prefix, filename)))
X = X.T

# create adata object for ATAC
adata = ad.AnnData(X=X, obs=pd.DataFrame(index=obs_bc), var=genomic_region_df)
adata.write(osp.join(data_path_prefix, "atac_count.h5ad"))


'''
Preprocess RNA data
'''
filename = "raw_data/GSM4156610_brain.rna.counts.txt.gz"
adata = ad.read_text(osp.join(data_path_prefix, filename))
adata = adata.transpose()

X = csc_matrix(adata.X)

obs = list(adata.obs.index)
rna_obs_bc = []
for n in obs:
    n = n.replace(",", ".")
    rna_obs_bc.append(n)

adata = ad.AnnData(X=X, obs=pd.DataFrame(index=rna_obs_bc), var=adata.var)
adata.write(osp.join(data_path_prefix, "rna_count.h5ad"))

del adata, X

'''
Preprocess cell types
'''
# read cell types, rna.bc and atac.bc
filename = "raw_data/GSM4156599_brain_celltype.txt.gz"
f = gzip.open(osp.join(data_path_prefix, filename))
temp = [line.strip().split(b"\t") for line in f.readlines()]
f.close()

temp = list(map(list, zip(*temp))) # transpose list of lists
atac_bc = [x.decode("utf-8")   for x in temp[0][1:]]
rna_bc = [x.decode("utf-8")  for x in temp[1][1:]]
cell_type = [x.decode("utf-8")  for x in temp[2][1:]]

rna2atac = dict(zip(rna_bc, atac_bc))
rna2celltype = dict(zip(rna_bc, cell_type))

'''
Load RNA and ATAC AnnDatas
Load sample barcodes
Order AnnDatas according to ordering of cell type labels
'''

filename = osp.join(data_path_prefix, "rna_count.h5ad")
adata_gex = ad.read_h5ad(filename)
barcode_gex = list(adata_gex.obs.index)

filename = osp.join(data_path_prefix, "atac_count.h5ad")
adata_atac = ad.read_h5ad(filename)

filename = "raw_data/GSM4156599_brain.barcodes.txt"
f = open(osp.join(data_path_prefix, filename))
# TODO Validate that the contents of obs_bc corresponds to RNA barcodes
obs_bc = [line.strip() for line in f.readlines()]
f.close()

# find order of mutual samples
FLAG = []
ATAC_barcode_new = []
GEX_barcode_new = []
Cell_type_new = []
for sample in obs_bc:
    FLAG.append(barcode_gex.index(sample))
    ATAC_barcode_new.append(rna2atac[sample])
    GEX_barcode_new.append(sample)
    Cell_type_new.append(rna2celltype[sample])

# slice rna data
X = adata_gex.X[np.array(FLAG)]
var_name = list(adata_gex.var.index)
# TODO does this really re-order rows according to the new index provided as `obs`
adata = ad.AnnData(X=X, obs=pd.DataFrame(index=GEX_barcode_new), var=pd.DataFrame(index=var_name))
adata.obs['cell_type'] = Cell_type_new

filename = osp.join(data_path_prefix,\
    "rna_count.preprocessed.h5ad")
adata.write(filename)

# slice atac data
X = adata_atac.X
var_name = list(adata_atac.var.index)
# TODO does this really re-order rows according to the new index provided as `obs`
adata = ad.AnnData(X=X, obs=pd.DataFrame(index=ATAC_barcode_new), var=adata_atac.var)
adata.obs['cell_type'] = Cell_type_new

filename = osp.join(data_path_prefix,\
    "atac_count.preprocessed.h5ad")
adata.write(filename)