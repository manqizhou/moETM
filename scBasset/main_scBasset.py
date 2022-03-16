# change working directory to parent
import os
os.chdir("../")

import pdb

# for loading functions and classes from model and training related files
import sys 
sys.path.insert(1,"")
sys.path.insert(2,"./scBasset/scBasset")

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import load_Buenrostro2018_dataset, prepare_Buenrostro2018_dataset
from scBasset.train import train_scBasset_AUC, train_scBasset


'''
Configuration
'''

dataset_path = f"./data/Buenrostro2018/preprocessed.Buenrostro2018.atac_ad.h5ad"
genome_dir = "./data/hg19" # validated from paper

# random state for weight initialization and dataset shuffling
seed = 35

# training on GPU/CPU
# if you want to train the model on GPU (recommended) 
# set `gpu` to True and set gpu_id
gpu = True
gpu_id = 1

# saving related 
dataset = "buenrostro2018"
result_dir_prefix = f"./experiments/scBasset/{dataset}/results"
model_dir_prefix = f"./experiments/scBasset/{dataset}/models"

# appended to the path on result_dir
eval_figures_suffix = "eval_figs"

# use training AUC or validation loss as early stopping measure 
monitor_AUC = True

'''
Load dataset
'''
adata_atac = load_Buenrostro2018_dataset(dataset_path, genome_dir=genome_dir,\
    seqlen=1344, verbose=True)
adata_atac = prepare_Buenrostro2018_dataset(adata_atac)

'''
Set model hyperparameters
'''

# device
if torch.cuda.is_available() and gpu:
    device = torch.device(f"cuda:{gpu_id}")
    print(f"> GPU {gpu_id} is available...")
else:
    device = torch.device("cpu")
    print("> GPU is not available... Switching to CPU")

model_hps = {}
model_hps["seq_embed_dim"] = 128
model_hps["seq_shift_max"] = 3
model_hps["seq_offset"] = 30
model_hps["first_conv_out_filters"] = 288
model_hps["first_conv_kernel_size"] = 17
model_hps["first_conv_pool_size"] = 3
model_hps["tower_conv_out_filters"] = 512
model_hps["tower_conv_kernel_size"] = 5
model_hps["tower_conv_pool_size"] = 2
model_hps["tower_conv_repeat"] = 6 
model_hps["channel_conv_out_filters"] = 256
model_hps["weight_init"] = "kaiming_normal"

'''
Set training parameters
'''

train_hps = {}
train_hps["eta"] = 3e-4
train_hps["bs"] = 256
train_hps["opt"] = "adam"
train_hps["max_epoch"] = 1000
train_hps["es_p"] = 50
train_hps["wd"] = 0
train_hps["device"] = device
train_hps["seed"] = seed
train_hps["summary"] = True

if monitor_AUC:
    monitor = "train-macro-AUC"
else:
    monitor = "vald-loss"

# update result and model dirs with respect to model hyperparameters
model_name = f"{monitor}_seed{train_hps['seed']}_bneck{model_hps['seq_embed_dim']}_wd{train_hps['wd']}_lr{train_hps['eta']}"
result_dir = f"{result_dir_prefix}/{model_name}"
model_dir = f"{model_dir_prefix}/{model_name}"

train_hps["model_dir"] = model_dir
train_hps["result_dir"] = result_dir

'''
Set evaluation parameters
'''
eval_params = {}
eval_params['batch_col'] = 'batch_indices'
eval_params['plot_fname'] = 'moETM'
eval_params['cell_type_col'] = 'cell_type'
eval_params['clustering_method'] = 'louvain'
eval_params['resolutions'] = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]
eval_params['plot_dir'] = f'{train_hps["result_dir"]}/eval_figs'

# final checks
os.makedirs(train_hps["model_dir"], exist_ok=True)
os.makedirs(train_hps["result_dir"], exist_ok=True)
os.makedirs(eval_params['plot_dir'], exist_ok=True)

# train the model
if monitor_AUC:
    train_scBasset_AUC(model_hps, train_hps, adata_atac, eval_params)
else:
    train_scBasset(model_hps, train_hps, adata_atac, eval_params)