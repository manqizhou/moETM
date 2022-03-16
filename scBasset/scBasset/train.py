import pdb
import time
import random
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchinfo import summary
from torch_utils import GenomeDataset, EarlyStopping

from build_model import scBasset

#######################################
## For debugging purposes, remove later
#######################################
import sys
sys.path.insert(1,"../")
from dataloader import load_mouse_brain_dataset, prepare_mouse_brain_dataset
from eval_utils import plot_loss, calculate_micro_metrics, calculate_macro_metrics, plot_multilabel_auroc, plot_multilabel_aupr, evaluate, plot_multilabel_macro_auroc, plot_multilabel_macro_aupr



def train_scBasset(model_hps, train_hps, adata_atac, Eval_kwargs, verbose=True):
    '''Training function for scBasset
    Args:
        model_hps: a dictionary which contains scBasset architecture related hyperparameters
            - seq_embed_dim:                the dimension of genome sequence embedding
            - seq_shift_max:                maximum shift amount for stochastic genome sequence shifting data augmentation
            - seq_offset:                   preprocessed genome sequences have `seq_offset` extra bases at each end
            - first_conv_out_filters:       the number of filters of the convolution layer directly operating on genome sequence
            - first_conv_kernel_size:       size of the convolution kernels operating on genome sequence
            - first_conv_pool_size:         size of the pooling employed after convolution on sequence
            - tower_conv_out_filters:       the number of output channels of the final convolution layer in tower
            - tower_conv_kernel_size:       size of the convolution kernels in convolution tower
            - tower_conv_repeat:            the number of convolution layers in the tower
            - channel_conv_out_filters:     out channel size of the 1x1 convolution layer after the tower
            - weight_init:                  weight initialization strategy for all layers in the network
        train_hps: a dictionary which contains training related hyperparameters
            - eta: learning rate
            - bs: batch_size
            - opt: optimizer type as string (e.g. `adam`) NOTE: need to change implementation for adding custom optimizers
            - max_epoch: max epoch count
            - es_p: early stopper patience
            - wd: weight decay
            - device: computing device in torch format
            - seed: random state for model weights and train test splits
            - summary: bool value to whether print architecture summary or not
        adata_mod2: preprocessed AnnData object for ATAC-seq data
        Eval_kwargs: same as moETM implementation
    '''

    # set random state for all available sources for reproducibility
    np.random.seed(train_hps["seed"])
    random.seed(train_hps["seed"])
    torch.manual_seed(train_hps["seed"])
    torch.cuda.manual_seed(train_hps["seed"])
    ## torch.backends.cudnn.deterministic = True
    ## torch.backends.cudnn.benchmark = False

    #######################################
    ## For debugging purposes, remove later
    #######################################
    # adata_atac = adata_atac[:10,:]

    num_cells = adata_atac.shape[0]

    # create model
    model = scBasset(num_cells, **model_hps)
    if train_hps["summary"]:
        print(summary(model, [train_hps["bs"], 1344 + 2 * model_hps["seq_offset"], 4]))
    model = model.to(train_hps["device"])

    # create training, validation and test
    num_peaks = adata_atac.shape[1]
    peak_idx = list(range(num_peaks))
    peak_y = [0] * num_peaks

    # # Option 1: split dataset for based on shuffling according to the provided random_state
    # # NOTE: This implementation can result in train or test splits to have cells that have only accessible labels
    # #       which causes error while calculation of AUC and AUPR
    train_peak_indices, vald_peak_indices, train_vald_peak_y, _ =\
        train_test_split(peak_idx, peak_y, test_size=0.2, random_state=train_hps["seed"])
    
    # create torch Dataset and DataLoader
    train_adata = adata_atac[:, train_peak_indices]
    vald_adata = adata_atac[:, vald_peak_indices]

    # all_dataset = GenomeDataset(adata_atac) # for peak-by-embedding matrix prediction purposes
    train_dataset = GenomeDataset(train_adata)
    vald_dataset = GenomeDataset(vald_adata)

    train_dataloader = DataLoader(train_dataset, batch_size=train_hps["bs"], shuffle=True)
    vald_dataloader = DataLoader(vald_dataset, batch_size=train_hps["bs"], shuffle=False)
    # all_dataloader = DataLoader(all_dataset, batch_size=train_hps["bs"], shuffle=False) # for peak-by-embedding matrix prediction purposes

    # create training constructs
    early_stopper = EarlyStopping(patience=train_hps["es_p"], verbose=True)
    # NOTE: BCELoss can be weighted
    loss_fn = nn.BCELoss()
    if train_hps["opt"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr = train_hps["eta"],
                                     weight_decay = train_hps["wd"],
                                     betas = (0.95, 0.9995) # TODO from scBasset paper determined by random search 
        )
    else:
        raise ValueError("Check optimizer type provided in train_hps")

    # training loop
    train_metrics = dict(AUC_micro=[], AUPR_micro=[], AUC_macro=[], AUPR_macro=[], loss=[])
    vald_metrics = dict(AUC_micro=[], AUPR_micro=[], AUC_macro=[], AUPR_macro=[], loss=[])
    current_result = None
    train_start = time.time()
    for epoch in range(1, train_hps["max_epoch"]+1):

        epoch_start = time.time()

        # train model
        train_loss = 0
        labels = []
        pred_probas = []
        for sequences, targets in train_dataloader:
            sequences = sequences.to(train_hps["device"])
            targets = targets.to(train_hps["device"])
            optimizer.zero_grad()
            preds = model(sequences)
            loss = loss_fn(preds, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # a candidate final epoch
            if train_hps["es_p"] - early_stopper.counter == 1 or train_hps["max_epoch"] - epoch <= 1:
                # record labels and predicted values
                labels.append(targets.detach().cpu().numpy())
                pred_probas.append(preds.detach().cpu().numpy())
        train_loss /= len(train_dataloader)
        train_metrics["loss"].append(train_loss)

        # a candidate final epoch
        if train_hps["es_p"] - early_stopper.counter == 1 or train_hps["max_epoch"] - epoch <= 1:
            # calculate micro/macro auc/aupr on training data
            labels = np.concatenate(labels, axis=0)
            pred_probas = np.concatenate(pred_probas, axis=0)
            # micro metrics
            micro_auroc, micro_aupr = calculate_micro_metrics(labels, pred_probas)
            train_metrics["AUC_micro"], train_metrics["AUPR_micro"] = micro_auroc, micro_aupr
            # macro metrics
            macro_auroc, macro_aupr = calculate_macro_metrics(labels, pred_probas)
            train_metrics["AUC_macro"], train_metrics["AUPR_macro"] = macro_auroc, macro_aupr

        # validate model on held out peaks
        vald_loss = 0
        labels = []
        pred_probas = []
        for sequences, targets in vald_dataloader:
            sequences = sequences.to(train_hps["device"])
            targets = targets.to(train_hps["device"])
            with torch.no_grad():
                preds = model(sequences)
            loss = loss_fn(preds, targets)
            vald_loss += loss.item()
            # a candidate final epoch
            if train_hps["es_p"] - early_stopper.counter == 1 or train_hps["max_epoch"] - epoch <= 1:
                # record labels and predicted values
                labels.append(targets.detach().cpu().numpy())
                pred_probas.append(preds.detach().cpu().numpy())
        vald_loss /= len(vald_dataloader)
        vald_metrics["loss"].append(vald_loss)
        

        # a candidate final epoch
        if train_hps["es_p"] - early_stopper.counter == 1 or train_hps["max_epoch"] - epoch <= 1:
            # calculate micro/macro auc/aupr on training data
            labels = np.concatenate(labels, axis=0)
            pred_probas = np.concatenate(pred_probas, axis=0)
            # micro metrics
            micro_auroc, micro_aupr = calculate_micro_metrics(labels, pred_probas)
            vald_metrics["AUC_micro"], vald_metrics["AUPR_micro"] = micro_auroc, micro_aupr
            # macro metrics
            macro_auroc, macro_aupr = calculate_macro_metrics(labels, pred_probas)
            vald_metrics["AUC_macro"], vald_metrics["AUPR_macro"] = macro_auroc, macro_aupr

        epoch_end = time.time()

        print(f"Epoch {epoch}, Train loss: {train_loss}, Vald loss: {vald_loss} (took {epoch_end - epoch_start} seconds)")

        # plot loss
        plot_fname = f"{Eval_kwargs['plot_dir']}/loss_plot.png"
        plot_title = f"scBasset, Epoch {epoch}"
        plot_loss(range(1, epoch+1), train_metrics["loss"], vald_metrics["loss"], plot_title, plot_fname)

        # # plot (micro, macro) x (AUC) x (train, vald)
        # plot_fname = f"{Eval_kwargs['plot_dir']}/auroc.png"
        # plot_title = f"scBasset, Epoch {epoch}, AUROC"
        # plot_multilabel_auroc(range(1, epoch+1), train_metrics["AUC_micro"], train_metrics["AUC_macro"],\
        #     vald_metrics["AUC_micro"], vald_metrics["AUC_macro"], plot_title, plot_fname)

        # # plot (micro, macro) x (AUPR) x (train, vald)
        # plot_fname = f"{Eval_kwargs['plot_dir']}/aupr.png"
        # plot_title = f"scBasset, Epoch {epoch}, AUPR"
        # plot_multilabel_aupr(range(1, epoch+1), train_metrics["AUPR_micro"], train_metrics["AUPR_macro"],\
        #     vald_metrics["AUPR_micro"], vald_metrics["AUPR_macro"], plot_title, plot_fname)

        # evaluate model according to cell type and batch separability of the embeddings
        embeddings = model.get_cell_embeddings().detach().cpu().numpy()
        adata_atac.obsm.update({"delta": embeddings})
        result = evaluate(adata=adata_atac, n_epoch=epoch, return_fig=False, **Eval_kwargs)
        print('>> Epoch %0d, Cell_ARI=%.4f, Cell_NMI=%.4f, Cell_ASW=%.4f, Cell_ASW2=%.4f' %
            (epoch, result['ari'], result['nmi'], result['asw'], result['asw_2']))

        current_result = result

        # save model if this is the best validation loss obtained until now
        if epoch != 1:
            if vald_loss <= min(vald_metrics["loss"]):
                savepath = f"{train_hps['model_dir']}/best_model.pth"
                # save model and optimizer
                torch.save({
                    "epoch":epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "latest_train_loss": train_loss,
                    "latest_vald_loss": vald_loss,
                    "train_hps": train_hps,
                    "model_hps": model_hps,
                    "eval_params": Eval_kwargs
                }, savepath)

        # early stoppping based on validation loss
        early_stopper(vald_loss)
        if early_stopper.stop:
            print(f"Early stopping at epoch {epoch}")
            break

    train_end = time.time()
    print(f"Training took {train_end - train_start} seconds.")

    # save calculated metrics
    results = dict(train_metrics=train_metrics, vald_metrics=vald_metrics, eval_results=current_result)
    with open(f"{train_hps['result_dir']}/metrics.pkl", "wb") as f:
        pickle.dump(results, f)


def train_scBasset_AUC(model_hps, train_hps, adata_atac, Eval_kwargs, verbose=True):
    '''Training function for scBasset
    Args:
        model_hps: a dictionary which contains scBasset architecture related hyperparameters
            - seq_embed_dim:                the dimension of genome sequence embedding
            - seq_shift_max:                maximum shift amount for stochastic genome sequence shifting data augmentation
            - seq_offset:                   preprocessed genome sequences have `seq_offset` extra bases at each end
            - first_conv_out_filters:       the number of filters of the convolution layer directly operating on genome sequence
            - first_conv_kernel_size:       size of the convolution kernels operating on genome sequence
            - first_conv_pool_size:         size of the pooling employed after convolution on sequence
            - tower_conv_out_filters:       the number of output channels of the final convolution layer in tower
            - tower_conv_kernel_size:       size of the convolution kernels in convolution tower
            - tower_conv_repeat:            the number of convolution layers in the tower
            - channel_conv_out_filters:     out channel size of the 1x1 convolution layer after the tower
            - weight_init:                  weight initialization strategy for all layers in the network
        train_hps: a dictionary which contains training related hyperparameters
            - eta: learning rate
            - bs: batch_size
            - opt: optimizer type as string (e.g. `adam`) NOTE: need to change implementation for adding custom optimizers
            - max_epoch: max epoch count
            - es_p: early stopper patience
            - wd: weight decay
            - device: computing device in torch format
            - seed: random state for model weights and train test splits
            - summary: bool value to whether print architecture summary or not
        adata_mod2: preprocessed AnnData object for ATAC-seq data
        Eval_kwargs: same as moETM implementation
    '''

    # set random state for all available sources for reproducibility
    np.random.seed(train_hps["seed"])
    random.seed(train_hps["seed"])
    torch.manual_seed(train_hps["seed"])
    torch.cuda.manual_seed(train_hps["seed"])
    ## torch.backends.cudnn.deterministic = True
    ## torch.backends.cudnn.benchmark = False

    #######################################
    ## For debugging purposes, remove later
    #######################################
    # adata_atac = adata_atac[:10,:]

    num_cells = adata_atac.shape[0]

    # create model
    model = scBasset(num_cells, **model_hps)
    if train_hps["summary"]:
        print(summary(model, [train_hps["bs"], 1344 + 2 * model_hps["seq_offset"], 4]))
    model = model.to(train_hps["device"])

    # create training, validation and test (for train_test_split in sklearn.model_selection)
    num_peaks = adata_atac.shape[1]
    peak_idx = np.array(list(range(num_peaks)))
    peak_y = np.array([0] * num_peaks)

    # # Option 1: split dataset for based on shuffling according to the provided random_state
    # # NOTE: This implementation can result in train or test splits to have cells that have only accessible labels
    # #       which causes error while calculation of AUC and AUPR
    train_peak_indices, vald_peak_indices, train_vald_peak_y, _ =\
        train_test_split(peak_idx, peak_y, test_size=0.2, random_state=train_hps["seed"])


    # process all data to get actual targets
    all_dataset = GenomeDataset(adata_atac)
    targets = all_dataset.targets.copy()
    del all_dataset

    # train_peak_indices, vald_peak_indices, _, _ = train_test_split(peak_idx, peak_y, test_size=0.2, random_state=i)
    # val_targets = targets[vald_peak_indices, :]
    # train_targets = targets[train_peak_indices, :]
    # (val_targets.sum(axis=0) == 0).sum()
    # (train_targets.sum(axis=0) == 0).sum()
    # # for i in range(0,200): train_peak_indices, vald_peak_indices, _, _ = train_test_split(peak_idx, peak_y, test_size=0.2, random_state=i);val_targets = targets[vald_peak_indices, :];train_targets = targets[train_peak_indices, :];print(f"seed: {i}\t{(val_targets.sum(axis=0) == 0).sum() + (train_targets.sum(axis=0) == 0).sum()}")

    # create torch Dataset and DataLoader
    train_adata = adata_atac[:, train_peak_indices]
    vald_adata = adata_atac[:, vald_peak_indices]

    # all_dataset = GenomeDataset(adata_atac) # for peak-by-embedding matrix prediction purposes
    train_dataset = GenomeDataset(train_adata)
    vald_dataset = GenomeDataset(vald_adata)

    train_dataloader = DataLoader(train_dataset, batch_size=train_hps["bs"], shuffle=True)
    vald_dataloader = DataLoader(vald_dataset, batch_size=train_hps["bs"], shuffle=False)
    # all_dataloader = DataLoader(all_dataset, batch_size=train_hps["bs"], shuffle=False) # for peak-by-embedding matrix prediction purposes

    # create training constructs
    early_stopper = EarlyStopping(patience=train_hps["es_p"], higher_is_better=True, verbose=True)
    # NOTE: BCELoss can be weighted
    loss_fn = nn.BCELoss()
    if train_hps["opt"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr = train_hps["eta"],
                                     weight_decay = train_hps["wd"],
                                     betas = (0.95, 0.9995) # TODO from scBasset paper determined by random search 
        )
    else:
        raise ValueError("Check optimizer type provided in train_hps")

    # training loop
    train_metrics = dict(AUC_micro=[], AUPR_micro=[], AUC_macro=[], AUPR_macro=[], loss=[])
    vald_metrics = dict(AUC_micro=[], AUPR_micro=[], AUC_macro=[], AUPR_macro=[], loss=[])
    current_result = None
    train_start = time.time()
    for epoch in range(1, train_hps["max_epoch"]+1):

        epoch_start = time.time()

        # train model
        train_loss = 0
        labels = []
        pred_probas = []
        for sequences, targets in train_dataloader:
            sequences = sequences.to(train_hps["device"])
            targets = targets.to(train_hps["device"])
            optimizer.zero_grad()
            preds = model(sequences)
            loss = loss_fn(preds, targets)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            # record labels and predicted values
            labels.append(targets.detach().cpu().numpy())
            pred_probas.append(preds.detach().cpu().numpy())
        train_loss /= len(train_dataloader)
        train_metrics["loss"].append(train_loss)

        # calculate micro/macro auc/aupr on training data
        labels = np.concatenate(labels, axis=0)
        pred_probas = np.concatenate(pred_probas, axis=0)
        # # micro metrics
        # micro_auroc, micro_aupr = calculate_micro_metrics(labels, pred_probas)
        # train_metrics["AUC_micro"].append(micro_auroc)
        # train_metrics["AUPR_micro"].append(micro_aupr)
        # macro metrics
        macro_auroc, macro_aupr = calculate_macro_metrics(labels, pred_probas)
        train_metrics["AUC_macro"].append(macro_auroc)
        train_metrics["AUPR_macro"].append(macro_aupr)

        # validate model on held out peaks
        vald_loss = 0
        labels = []
        pred_probas = []
        for sequences, targets in vald_dataloader:
            sequences = sequences.to(train_hps["device"])
            targets = targets.to(train_hps["device"])
            with torch.no_grad():
                preds = model(sequences)
            loss = loss_fn(preds, targets)
            vald_loss += loss.item()
            # record labels and predicted values
            labels.append(targets.detach().cpu().numpy())
            pred_probas.append(preds.detach().cpu().numpy())
        vald_loss /= len(vald_dataloader)
        vald_metrics["loss"].append(vald_loss)
        

        # calculate micro/macro auc/aupr on training data
        labels = np.concatenate(labels, axis=0)
        pred_probas = np.concatenate(pred_probas, axis=0)
        # # micro metrics
        # micro_auroc, micro_aupr = calculate_micro_metrics(labels, pred_probas)
        # vald_metrics["AUC_micro"].append(micro_auroc)
        # vald_metrics["AUPR_micro"].append(micro_aupr)
        # macro metrics
        macro_auroc, macro_aupr = calculate_macro_metrics(labels, pred_probas)
        vald_metrics["AUC_macro"].append(macro_auroc)
        vald_metrics["AUPR_macro"].append(macro_aupr)

        epoch_end = time.time()

        print(f"Epoch {epoch}, Train loss: {train_loss}, Vald loss: {vald_loss} (took {epoch_end - epoch_start} seconds)")

        # plot loss
        plot_fname = f"{Eval_kwargs['plot_dir']}/loss_plot.png"
        plot_title = f"scBasset, Epoch {epoch}"
        plot_loss(range(1, epoch+1), train_metrics["loss"], vald_metrics["loss"], plot_title, plot_fname)

        # plot (micro, macro) x (AUC) x (train, vald)
        plot_fname = f"{Eval_kwargs['plot_dir']}/auroc.png"
        plot_title = f"scBasset, Epoch {epoch}, AUROC"
        plot_multilabel_macro_auroc(range(1, epoch+1), train_metrics["AUC_macro"], vald_metrics["AUC_macro"], plot_title, plot_fname)

        # plot (micro, macro) x (AUPR) x (train, vald)
        plot_fname = f"{Eval_kwargs['plot_dir']}/aupr.png"
        plot_title = f"scBasset, Epoch {epoch}, AUPR"
        plot_multilabel_macro_aupr(range(1, epoch+1), train_metrics["AUPR_macro"], vald_metrics["AUPR_macro"], plot_title, plot_fname)

        # evaluate model according to cell type and batch separability of the embeddings
        embeddings = model.get_cell_embeddings().detach().cpu().numpy()
        adata_atac.obsm.update({"delta": embeddings})
        result = evaluate(adata=adata_atac, n_epoch=epoch, return_fig=False, **Eval_kwargs)
        print('>> Epoch %0d, Cell_ARI=%.4f, Cell_NMI=%.4f, Cell_ASW=%.4f, Cell_ASW2=%.4f' %
            (epoch, result['ari'], result['nmi'], result['asw'], result['asw_2']))

        current_result = result

        # save model if this vald loss started to increase
        if early_stopper.counter >= 1:
            savepath = f"{train_hps['model_dir']}/best_model.pth"
            # save model and optimizer
            torch.save({
                "epoch":epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "latest_train_loss": train_loss,
                "latest_vald_loss": vald_loss,
                "latest_train_macro_AUC": train_metrics["AUC_macro"][-1],
                "latest_vald_macro_AUC": vald_metrics["AUC_macro"][-1],
                "train_hps": train_hps,
                "model_hps": model_hps,
                "eval_params": Eval_kwargs
            }, savepath)

        # early stoppping based on validation loss
        early_stopper(train_metrics["AUC_macro"][-1])
        if early_stopper.stop:
            print(f"Early stopping at epoch {epoch}")
            break

    train_end = time.time()
    print(f"Training took {train_end - train_start} seconds.")

    # save calculated metrics
    results = dict(train_metrics=train_metrics, vald_metrics=vald_metrics, eval_results=current_result)
    with open(f"{train_hps['result_dir']}/metrics.pkl", "wb") as f:
        pickle.dump(results, f)
