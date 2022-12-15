

from moETM.train import Trainer_moETM_for_cross_prediction, Train_moETM_for_cross_prediction
from dataloader import load_nips_rna_atac_dataset, prepare_nips_dataset, data_process_moETM_cross_prediction
from moETM.build_model import build_moETM
import pandas as pd
import gc
import os
import numpy as np
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

n_total_sample = adata_mod1.shape[0]

X_mod1_train_T, X_mod2_train_T, batch_index_train_T, X_mod1_test_T, X_mod2_test_T, batch_index_test_T, test_adata_mod1, train_adata_mod1, test_mod1_sum, test_mod2_sum= data_process_moETM_cross_prediction(adata_mod1, adata_mod2, n_sample=np.int(np.floor(n_total_sample*0.8)))

num_batch = len(batch_index_train_T.unique())
input_dim_mod1 = X_mod1_train_T.shape[1]
input_dim_mod2 = X_mod2_train_T.shape[1]
train_num = X_mod1_train_T.shape[0]

num_topic = 200
emd_dim = 400
encoder_mod1, encoder_mod2, decoder, optimizer = build_moETM(input_dim_mod1, input_dim_mod2, num_batch, num_topic=num_topic, emd_dim=emd_dim)

direction = 'rna_to_another'   # Or another_to_rna
trainer = Trainer_moETM_for_cross_prediction(encoder_mod1, encoder_mod2, decoder, optimizer, direction)

Total_epoch = 500
batch_size = 2000
Train_set = [X_mod1_train_T, X_mod2_train_T, batch_index_train_T]
Test_set = [X_mod1_test_T, X_mod2_test_T, batch_index_test_T, test_adata_mod1, test_mod1_sum, test_mod2_sum]
Train_moETM_for_cross_prediction(trainer, Total_epoch, train_num, batch_size, Train_set, Test_set)

