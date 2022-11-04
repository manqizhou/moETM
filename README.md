## Learning single-cell multi-omic signature of gene regulatory programs by deep embedding topic model

Please contact (mz335@cornell.edu or xxx) if you have any questions or suggestions.

![model](./model.png?raw=true "Title")

---
## Installation
Git clone a copy of code:
```
git clone https://github.com/manqizhou/moETM.git
```
## Required dependencies

* [python](https://www.python.org) (3.9)
* [PyTorch](https://pytorch.org/) (0.4.1)
* [scanpy](https://scanpy.readthedocs.io/en/stable/) (1.9)
* [anndata](https://anndata.readthedocs.io/en/latest/) (0.9)

## Dataset
* [BMMC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122) The Bone Marrow Mononuclear Cell datasets from 2021 NeurIPS challenge.
* [MSLAC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203) The Mouse Skin Late Anagen Cell dataset.
* [MKC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117089) The Mouse Kidney Cell dataset.
* [MBC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203)  The Mouse Brain Cell dataset.
* [HWBC](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378) The Human White Blood Cell dataset.
* [HBIC](https://www.nature.com/articles/s41591-021-01329-2#data-availability) The Human Blood Immune Cell dataset.


## Run
1. Please run the main function: main_moETM.py

2. I uploaded two datasets: BMMC(nips) and mouse_skin(MSLAC)
NIPS dataset has multiple batches, so moETM has bias correction term. 
MS dataset has 1 batch, so moETM do not have correction term.

