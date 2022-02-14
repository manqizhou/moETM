import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import anndata
import os
import seaborn as sns
from plotnine import *

from scvi.dataset import AnnDatasetFromAnnData, CellMeasurement, GeneExpressionDataset
from scvi.models import TOTALVI
from scvi.inference import TotalPosterior, TotalTrainer
from scvi import set_seed

set_seed(0)

sc.set_figure_params(figsize=(4, 4))

pbmc_10k_adata = sc.read("pbmc_10k_protein_v3.h5ad", backup_url="https://github.com/YosefLab/scVI-data/raw/master/pbmc_10k_protein_v3.h5ad")
pbmc_5k_adata = sc.read("pbmc_5k_protein_v3.h5ad", backup_url="https://github.com/YosefLab/scVI-data/raw/master/pbmc_5k_protein_v3.h5ad")

dataset_10k = AnnDatasetFromAnnData(pbmc_10k_adata, cell_measurements_col_mappings={"protein_expression":"protein_names"})
dataset_5k = AnnDatasetFromAnnData(pbmc_5k_adata, cell_measurements_col_mappings={"protein_expression":"protein_names"})

print(dataset_10k.protein_names.shape)
print(dataset_5k.protein_names.shape)

dataset = GeneExpressionDataset()
dataset.populate_from_datasets([dataset_10k, dataset_5k])
dataset.subsample_genes(4000)

print("dataset.batch_indices")
print(dataset.batch_indices)

dataset.protein_expression[dataset.batch_indices.ravel() == 1] = np.zeros_like(dataset.protein_expression[dataset.batch_indices.ravel() == 1])
# dataset_5k was modified in place during the combination step
assert np.array_equal(dataset_5k.protein_names, dataset.protein_names)
held_out_proteins = dataset_5k.protein_expression
print("dataset_5k.protein_expression[:5]")

protein_batch_mask = dataset.get_batch_mask_cell_measurement("protein_expression")

print("protein_batch_mask")
print(protein_batch_mask)

print("dataset.n_batches")
print(dataset.n_batches)

totalvae = TOTALVI(dataset.nb_genes,
                   len(dataset.protein_names),
                   n_batch=dataset.n_batches,
                   protein_batch_mask=protein_batch_mask
)
use_cuda = True
lr = 4e-3
n_epochs = 500

trainer = TotalTrainer(
    totalvae,
    dataset,
    train_size=0.90,
    test_size=0.10,
    use_cuda=use_cuda,
    frequency=1,
    batch_size=256,
    use_adversarial_loss=True
)

trainer.train(lr=lr, n_epochs=n_epochs)

plt.plot(trainer.history["elbo_train_set"], label="train")
plt.plot(trainer.history["elbo_test_set"], label="test")
plt.title("Negative ELBO over training epochs")
plt.ylim(1100, 1500)
plt.legend()

# create posterior on full data
full_posterior = trainer.create_posterior(type_class=TotalPosterior)
# update the batch size for GPU operations on Colab
full_posterior = full_posterior.update({"batch_size":32})

# extract latent space
latent_mean = full_posterior.get_latent()[0]
# NOTE: latent_mean is the latent representation later on for clustering.
print("Shape of latent_mean")
print(latent_mean.shape)
