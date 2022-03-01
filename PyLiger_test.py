import pyliger

from anndata import read_h5ad

data_dir = ''
ctrl_dge = read_h5ad(data_dir + 'pbmc_ctrl_downsampled.h5ad')
stim_dge = read_h5ad(data_dir + 'pbmc_stim_downsampled.h5ad')

# With the digital gene expression matrices for both datasets, we can initialize a pyliger object using create_liger function.
adata_list = [ctrl_dge, stim_dge]
ifnb_liger = pyliger.create_liger(adata_list)

# Before we can run iNMF on our datasets, we must run several preprocessing steps to normalize expression data to account for differences in sequencing depth and efficiency between cells, identify variably expressed genes, and scale the data so that each gene has the same variance. Note that because nonnegative matrix factorization requires positive values, we do not center the data by subtracting the mean. We also do not log transform the data.

pyliger.normalize(ifnb_liger)
pyliger.select_genes(ifnb_liger)
pyliger.scale_not_center(ifnb_liger)

print(ifnb_liger)

# We are now able to run integrative non-negative matrix factorization on the normalized and scaled datasets. The key parameter for this analysis is k, the number of matrix factors (analogous to the number of principal components in PCA). In general, we find that a value of k between 20 and 40 is suitable for most analyses and that results are robust for choice of k. Because LIGER is an unsupervised, exploratory approach, there is no single “right” value for k, and in practice, users choose k from a combination of biological prior knowledge and other information.
pyliger.optimize_ALS(ifnb_liger, k = 20)

print(ifnb_liger)

#We can now use the resulting factors to jointly cluster cells and perform quantile normalization by dataset, factor, and cluster to fully integrate the datasets. All of this functionality is encapsulated within the quantile_norm function, which uses max factor assignment followed by refinement using a k-nearest neighbors graph.
pyliger.quantile_norm(ifnb_liger)

# The quantile_norm procedure produces joint clustering assignments and a low-dimensional representation that integrates the datasets together. These joint clusters directly from iNMF can be used for downstream analyses (see below). Alternatively, you can also run Louvain community detection, an algorithm commonly used for single-cell data, on the normalized cell factors. The Louvain/Leiden algorithm excels at merging small clusters into broad cell classes and thus may be more desirable in some cases than the maximum factor assignments produced directly by iNMF.
pyliger.leiden_cluster(ifnb_liger, resolution=0.25)

# To visualize the clustering of cells graphically, we can project the normalized cell factors to two or three dimensions. Liger supports both t-SNE and UMAP for this purpose. Note that if both techniques are run, the object will only hold the results from the most recent.
#pyliger.run_umap(ifnb_liger, distance = 'cosine', n_neighbors = 30, min_dist = 0.3)

# plot_by_dataset_and_cluster returns two graphs, generated by t-SNE or UMAP in the previous step. The first colors cells by dataset of origin, and the second by cluster as determined by Liger. The plots provide visual confirmation that the datasets are well aligned and the clusters are consistent with the shape of the data as revealed by UMAP.
#%matplotlib notebook
#all_plots = pyliger.plot_by_dataset_and_cluster(ifnb_liger, axis_labels = ['UMAP 1', 'UMAP 2'], return_plots = True)
#all_plots

#PRF1 = pyliger.plot_gene(ifnb_liger, "PRF1", axis_labels = ['UMAP 1', 'UMAP 2'], return_plots = True)
#PRF1

# Using the run_wilcoxon function, we can next identify gene markers for all clusters. We can also compare expression within each cluster across datasets, which in this case reveals markers of interferon-beta stimulation. The function returns a table of data that allows us to determine the significance of each gene’s differential expression, including log fold change, area under the curve and p-value.

#cluster_results = pyliger.run_wilcoxon(ifnb_liger, compare_method = "clusters")

#cluster_results

