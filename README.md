# moETM: Learning single-cell multi-omic signature of gene regulatory programs by deep embedding topic model

moETM integrates multiomics data across different experiments or studies with interpretable latent embeddings.

Please contact (mz335@cornell.edu or zhanghao.duke@gmail.com or yueli@cs.mcgill.ca or few2001@med.cornell.edu) if you have any questions or suggestions.


## Contents ##

- [Model Overview](#model-overview)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
	- [Data Preparation](#data-preparation)
	- [Integration](#moetm)
	- [Imputation](#imputation)
	- [Inclusion of prior pathway knowledge](#inclusion-of-prior-pathway-knowledge)
- [Downstream analysis](#downstream-analysis)

## Model Overview

moETM generalizes the widely used variational autoencoder (VAE) to model multi-modal data. Specifically, the encoder in moETM is a fully-connected neural network (NN), which infers topic proportion from multi-omic normalized count vectors for a cell. The decoder in moETM is a linear multi-modal ETM reconstructing the normalized count vectors from the latent topic proportion. The parameters in moETM are learned by maximizing the evidence lower bound (ELBO) of the marginal data likelihood under the framework of amortized variational inference

![model](./model.png?raw=true "Title")

a) The moETM model. The left panel represents different modalities. The medium panel represents the fully-connected neural network encoder. Before inputting values into the encoder, column normalization was applied for each modality. The right panel is the linear decoder.
b) The moETM clustering process. The topic proportion $\theta$ from the trained encoder can be clustered and visualized in 2D dimension.
c) Cross-modality imputation. The missing modality can be imputed using the topic embedding and topic matrix from the reference trained decoder. 
d) Downstream qualitative analysis. Transcriptomics, epigenetic, and protein signatures from the trained feature-by-topic matrix and enriched cell types from the trained cell-by-topic matrix can be used for enrichment analysis, embedding visualization, and exploring topic-directed regulatory networks.


## Installation

Git clone a copy of code:
```
git clone https://github.com/manqizhou/moETM.git
```
moETM requires several dependencies:

* [python](https://www.python.org) 
* [PyTorch](https://pytorch.org/) 
* [scanpy](https://scanpy.readthedocs.io/en/stable/) 
* [anndata](https://anndata.readthedocs.io/en/latest/) 


## Datasets

There were 7 public datasets included in this study for performance evaluation and model comparison. All the 7 datasets are from publicly available repositories. Among them, 4 datatsets (BMMC1, MSLAC, MKC, MBC) consist of gene expression and chromatin accessibility information and 3 datasets (BMMC2, HWBC, HBIC) include gene expression and surface protein information. Sepcifically, the HBIC dataset was measured from both COVID patients and healthy patients. 

* [BMMC1 and BMMC2](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122) The Bone Marrow Mononuclear Cell datasets from 2021 NeurIPS challenge.
* [MSLAC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203) The Mouse Skin Late Anagen Cell dataset.
* [MKC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117089) The Mouse Kidney Cell dataset.
* [MBC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203)  The Mouse Brain Cell dataset.
* [HWBC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) The Human White Blood Cell dataset.
* [HBIC](https://www.nature.com/articles/s41591-021-01329-2#data-availability) The Human Blood Immune Cell dataset.


## Usage

### Data preparation

moETM requires cell-by-feature matrices as input, where feature could be gene, protein, or peak. The input data is in `AnnData` format and is loaded and preprocessed by the `load_*_dataset()` and `prepare_*_dataset()` functions in the `dataloader.py` script. Before putting into the model, all matrices are column normalized by dividing the column sum.

### Integration

Please run the main script `main_integration_.py` and edit data path accordingly.

For the gene + protein case, please refer to [`main_integration_rna_protein.py`](/main_integration_rna_protein.py) for details. 
For the gene + peak case, please refer to [`main_integration_rna_atac.py`](/main_integration_rna_atac.py) for details.

### Imputation 

please refer to [`main_cross_prediction_rna_atac.py`](/main_cross_prediction_rna_atac.py) and [`main_cross_prediction_rna_protein.py`](/main_cross_prediction_rna_protein.py) for details. The two scripts are the same during training but different in the data preparation part.

### Inclusion of prior pathway knowledge

moETM can use prior pathway knoeledge information by adding a pathway-by-gene matrix in the encoder. We downloaded pathways from [MSgiDB](https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp), and selected the C7: immunologic signature gene sets. We kept pathways that contain more than 5 and fewer than 100 genes.

Please refer to [`main_integration_rna_atac_use_pathway.py`](/main_integration_rna_atac_use_pathway.py) for details.

## Downstream analysis

Scripts that are used to do downstream analysis or plotting are included in the [`downstream_analysis`](/downstream_analysis) folder.

