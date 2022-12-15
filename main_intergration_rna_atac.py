

from moETM.train import Trainer_moETM, Train_moETM
from dataloader import load_nips_rna_atac_dataset, prepare_nips_dataset, data_process_moETM
from moETM.build_model import build_moETM
import pandas as pd
import gc
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import warnings
warnings.filterwarnings('ignore')

# Load dataset
mod_file_path = "./data/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad"
gene_encoding = pd.read_csv('./useful_file/gene_coding_nips_rna_atac.csv')

adata_mod1, adata_mod2 = load_nips_rna_atac_dataset(mod_file_path, gene_encoding)
gc.collect()

# Prepare dataset
adata_mod1, adata_mod2 = prepare_nips_dataset(adata_mod1, adata_mod2)

# Evaluation parameters
Eval_kwargs = {}
Eval_kwargs['batch_col'] = 'batch_indices'
Eval_kwargs['plot_fname'] = 'moETM_delta'
Eval_kwargs['cell_type_col'] = 'cell_type'
Eval_kwargs['clustering_method'] = 'louvain'
Eval_kwargs['resolutions'] = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
Eval_kwargs['plot_dir'] = './result_fig'

n_total_sample = adata_mod1.shape[0]

X_mod1_train_T, X_mod2_train_T, batch_index_train_T, train_adata_mod1 = data_process_moETM(adata_mod1, adata_mod2)

num_batch = len(batch_index_train_T.unique())
input_dim_mod1 = X_mod1_train_T.shape[1]
input_dim_mod2 = X_mod2_train_T.shape[1]
train_num = X_mod1_train_T.shape[0]

num_topic = 100
emd_dim = 400
encoder_mod1, encoder_mod2, decoder, optimizer = build_moETM(input_dim_mod1, input_dim_mod2, num_batch, num_topic=num_topic, emd_dim=emd_dim)

trainer = Trainer_moETM(encoder_mod1, encoder_mod2, decoder, optimizer)

Total_epoch = 500
batch_size = 2000
Train_set = [X_mod1_train_T, X_mod2_train_T, batch_index_train_T]
Test_set = [X_mod1_train_T, X_mod2_train_T, batch_index_train_T, train_adata_mod1]
Train_moETM(trainer, Total_epoch, train_num, batch_size, Train_set, Test_set, Eval_kwargs)

