import pdb
import os 

import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F


class StochasticReverseComplement(nn.Module):
    '''Stochastically reverse complement sequences in the batch
    while model training
    '''

    def __init__(self):
        super(StochasticReverseComplement, self).__init__()

    def forward(self, seq_1hot):
        if self.training:
            # reverse complement
            rc_seq_1hot = torch.index_select(seq_1hot, dim=-1,\
                index=torch.Tensor([3,2,1,0]).long().to(seq_1hot.device))
            rc_seq_1hot = torch.flip(rc_seq_1hot, dims=[-2])
            # stochastic counterpart
            temp = torch.rand(1).item()
            cond =  torch.full(seq_1hot.shape, temp).to(seq_1hot.device) > 0.5 # TODO should be moved to gpu
            cond_seq_1hot = torch.where(cond, rc_seq_1hot, seq_1hot)
            return cond_seq_1hot
        else:
            return seq_1hot


class StochasticShift(nn.Module):

    def __init__(self,
                 shift_max,
                 offset=30
    ):
        '''Stochastically shift one hot encoded sequences in the batch
        while model training. Assumes the input sequences to have 
        2 * [offset] + [peak_length] positions where the first [offset] values
        of the sequence at each side does not count as peak region.

        Args:
            shift_max: maximum shift amount for the sequences
            offset: The length of padding at each side of the DNA sequence
                added while data preparation
        '''
        super(StochasticShift, self).__init__()

        self.shift_max = shift_max
        self.offset = offset
        
    def forward(self, seq_1hot):
        if self.training:
            shift_amount = torch.randint(low=-self.shift_max, high=self.shift_max+1,\
                size = (1,)).item()
            # seq_1hot here should be either 3 or 4 dimensional
            if len(seq_1hot.shape) == 3:
                return seq_1hot[:, self.offset+shift_amount:-self.offset+shift_amount, :]
            else: 
                return seq_1hot[:, :, self.offset+shift_amount:-self.offset+shift_amount, :]
        else:
            # x here should be either 3 or 4 dimensional
            if len(seq_1hot.shape) == 3:
                return seq_1hot[:, self.offset:-self.offset, :]
            else:
                return seq_1hot[:, :, self.offset:-self.offset, :]


class Conv2dBlock(nn.Module):
    ''' A 2D convolution block to process [batch_size, sequence, features]
    shaped inputs.
    '''

    def __init__(self,
                 out_filter,
                 kernel_size=1,
                 pool_size=1,
                 bn=True,
                 weight_init="kaiming_normal"
    ):
        '''
        Args: 
            out_filter: number of output channels
            kernel_size: convolution kernel size
            pool_size: max pooling kernel size employed after convolution
            bn: whether or not to include batch norm after convolution
                and before pooling
            weight_init: weight initialization method (see below for the possible values)
        Returns:
            [batch_size, out_filter, features] output tensor
        '''

        super(Conv2dBlock, self).__init__()

        self.out_filter = out_filter
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        self.bn = nn.BatchNorm1d(self.out_filter) if bn else nn.Identity()

        self.conv = nn.Conv2d(
            in_channels = 1,
            out_channels = self.out_filter,
            kernel_size = (self.kernel_size, 4),
            stride = 1,
            padding = ((self.kernel_size-1)//2,0)
        )
        
        if weight_init == "xavier_normal":
            nn.init.xavier_normal_(self.conv.weight.data)
        elif weight_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.conv.weight.data)
        elif weight_init == "kaiming_normal":
            nn.init.kaiming_normal_(self.conv.weight.data)
        elif weight_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.conv.weight.data)
        else:
            raise ValueError("Please check weight initialization method for convolution kernels")

        self.gelu = nn.GELU()
        self.pool = nn.MaxPool1d(self.pool_size, padding=(self.pool_size-1)//2)
    
    def forward(self, _x):
        # # input to the model is (N,H,W)
        # # input format of Conv2d is (N,C,H,W)
        # # expand C dim
        x_ = _x.unsqueeze(1)
        x_ = self.conv(x_)
        # squeeze back to 1D sequence input
        x_ = x_.squeeze(-1)
        x_ = self.bn(x_)
        x_ = self.pool(x_)
        return x_


class Conv1dBlock(nn.Module):
    '''A 1D convolution block to process [batch_size, sequence, features]
    shaped inputs
    '''

    def __init__(self,
                 in_filter,
                 out_filter,
                 kernel_size=1,
                 pool_size=1,
                 bn=True,
                 dropout=0,
                 weight_init="kaiming_normal"
    ):
        '''
        Args: 
            in_filter: number of input channels
            out_filter: number of output channles
            kernel_size: convolution kernel size
            pool_size: max pooling kernel size employed after convolution
            bn: whether or not to include batch norm after convolution
                and before pooling
            dropout: dropout probability employed after batch norm
                and before pooling
            weight_init: weight initialization for convolution
        Returns:
            [batch_size, out_filter, features] output tensor
        '''

        super(Conv1dBlock, self).__init__()

        self.in_filter = in_filter
        self.out_filter = out_filter
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        self.bn = nn.BatchNorm1d(self.out_filter) if bn else nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.conv = nn.Conv1d(
            in_channels = self.in_filter,
            out_channels = self.out_filter,
            kernel_size = self.kernel_size,
            stride = 1,
            padding = "same"
        )

        if weight_init == "xavier_normal":
            nn.init.xavier_normal_(self.conv.weight.data)
        elif weight_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.conv.weight.data)
        elif weight_init == "kaiming_normal":
            nn.init.kaiming_normal_(self.conv.weight.data)
        elif weight_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.conv.weight.data)
        else:
            raise ValueError("Please check weight initialization method for convolution kernels")

        self.gelu = nn.GELU()
        self.pool = nn.MaxPool1d(self.pool_size, padding=int((self.pool_size-1)/2))

    def forward(self, _x):
        x_ = self.gelu(_x)
        x_ = self.conv(x_)
        x_ = self.bn(x_)
        x_ = self.dropout(x_)
        x_ = self.pool(x_)
        return x_


class Conv1dTower(nn.Module):
    def __init__(self,
                 in_filter_init,
                 filter_mult=None,
                 out_filter_end=None,
                 divisible_by=1,
                 repeat=2,
                 **kwargs
    ):
        '''1D convolution tower block of specified depth
        Args:
            in_filter_init: Initial conv filter size
            filter_mult:    Multiplier for conv filter sizes
            out_filter_end: Final conv filter size 
            divisible_by:   Round filters to be divisible by (e.g. a power of two)
            repeat:         Tower repetitions (NOTE: repeat should not be 1 at any case)
        Returns:
            [batch_size, out_filter_end, *] shaped output tensor
        '''
        super(Conv1dTower, self).__init__()

        self.in_filter_init = in_filter_init
        self.filter_mult = filter_mult
        self.out_filter_end = out_filter_end
        self.divisible_by = divisible_by
        self.repeat = repeat

        self._round = lambda x: int(np.round(x / divisible_by) * divisible_by)

        if self.filter_mult is None:
            assert self.out_filter_end is not None
            self.filter_mult = np.exp(np.log(self.out_filter_end / self.in_filter_init) / (self.repeat - 1))

        # create convolution tower blocks
        modules = []
        repeat_filter = self.in_filter_init
        # first conv retains size
        modules.append(Conv1dBlock(
                repeat_filter,
                repeat_filter,
                **kwargs
            ))
        for r in range(repeat-1):
            out_filter = self._round(repeat_filter * self.filter_mult)
            modules.append(Conv1dBlock(
                repeat_filter,
                out_filter,
                **kwargs
            ))
            repeat_filter = out_filter

        self.tower = nn.Sequential(*modules) 

    def forward(self, _x):
        x_ = self.tower(_x)
        return x_  


class DenseBlock(nn.Module):
    '''
    Fully connected layer blocks which includes batch normalization and dropout
    '''

    def __init__(self, 
                 in_size,
                 out_size,
                 bias=False,
                 flatten=True,
                 bn=True,
                 dropout=0,
                 weight_init="kaiming_normal"
    ):
        '''
        Args:
            in_size:        number of input neurons for the layer
            out_size:       number of output neurons for the layer
            bias:           whether to use bias
            flatten:        whether to flatten inputs before transformation
            bn:             whether to use batch normalization
            dropout:        dropout probability
            weight_init:    weight initializing method
        Returns:
            [batch_size, out_size] shaped output tensor
        '''
        super(DenseBlock, self).__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias

        self.gelu = nn.GELU()

        self.flatten = nn.Flatten(start_dim=1) if flatten else nn.Identity()

        self.bn = nn.BatchNorm1d(self.out_size) if bn else nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.fc = nn.Linear(self.in_size, self.out_size, bias=self.bias)

        if weight_init == "xavier_normal":
            nn.init.xavier_normal_(self.fc.weight.data)
            if self.bias:
                nn.init.constant_(self.fc.bias.data, 0.01)
        elif weight_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.fc.weight.data)
            if self.bias:
                nn.init.constant_(self.fc.bias.data, 0.01)
        elif weight_init == "kaiming_normal":
            nn.init.kaiming_normal_(self.fc.weight.data)
            if self.bias:
                nn.init.constant_(self.fc.bias.data, 0.01)
        elif weight_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.fc.weight.data)
            if self.bias:
                nn.init.constant_(self.fc.bias.data, 0.01)
        else:
            raise ValueError("Please check weight initialization method for dense block")

    def forward(self, _x):
        x_ = self.gelu(_x)
        x_ = self.flatten(x_)
        x_ = self.fc(x_)
        x_ = self.bn(x_)
        x_ = self.dropout(x_)
        return x_


class SequenceEncoder(nn.Module):
    '''
    Sequence encoder module executing following modules in order:
        StochasticReverseComplement
        StochasticShift
        Conv2dBlock
        Conv1dTower
        Conv1dBlock (1x1 conv)
        DenseBlock
    '''

    def __init__(self, 
                 seq_embed_dim=32,
                 first_conv_out_filters=288,
                 first_conv_kernel_size=17,
                 first_conv_pool_size=3,
                 tower_conv_out_filters=128*3,
                 tower_conv_kernel_size=5,
                 tower_conv_pool_size=2,
                 tower_conv_repeat=6, 
                 channel_conv_out_filters=128,
                 weight_init="kaiming_normal"
    ):
        '''
        Args:
            seq_embed_dim:              the dimension of genome sequence embedding
            first_conv_out_filters:     the number of filters of the convolution
                                        layer directly operating on genome sequence
            first_conv_kernel_size:     size of the convolution kernels
                                        operating on genome sequence
            first_conv_pool_size:       size of the pooling employed after
                                        convolution on sequence
            tower_conv_out_filters:     the number of output channels of the final
                                        convolution layer in tower
            tower_conv_kernel_size:     size of the convolution kernels
                                        in convolution tower
            tower_conv_repeat:          the number of convolution layers in the tower
            channel_conv_out_filters:   out channel size of the 1x1 convolution
                                        layer after the tower
            weight_init:                weight initialization strategy for all layers
                                        in the network
        Returns:
            [batch_size, ] chromatin accesibility probabilities
        '''
        super(SequenceEncoder, self).__init__()

        self.first_conv_out_filters = first_conv_out_filters
        self.first_conv_kernel_size = first_conv_kernel_size
        self.first_conv_pool_size = first_conv_pool_size
        self.tower_conv_out_filters = tower_conv_out_filters
        self.tower_conv_kernel_size = tower_conv_kernel_size
        self.tower_conv_pool_size = tower_conv_pool_size
        self.tower_conv_repeat = tower_conv_repeat
        self.channel_conv_out_filters = channel_conv_out_filters
        self.seq_embed_dim = seq_embed_dim
        self.weight_init = weight_init

        self.seq_conv = Conv2dBlock(out_filter=self.first_conv_out_filters,
                                    kernel_size=self.first_conv_kernel_size,
                                    pool_size=self.first_conv_pool_size)

        self.tower = Conv1dTower(in_filter_init=self.first_conv_out_filters,
                                 out_filter_end=self.tower_conv_out_filters,
                                 repeat=self.tower_conv_repeat,
                                 kernel_size=self.tower_conv_kernel_size,
                                 pool_size=self.tower_conv_pool_size)

        self.channel_conv = Conv1dBlock(in_filter=self.tower_conv_out_filters,
                                        out_filter=self.channel_conv_out_filters,
                                        kernel_size=1,
                                        pool_size=1)
        
        self.bottleneck = DenseBlock(in_size=self.channel_conv_out_filters*7, # NOTE: Mind this 7 which is the sequence depth after conv layers
                                     out_size=self.seq_embed_dim,
                                     dropout=0.2)
        
        self.gelu = nn.GELU()

    def forward(self, _x):
        x_ = self.seq_conv(_x)
        x_ = self.tower(x_)
        x_ = self.channel_conv(x_)
        x_ = self.bottleneck(x_)
        x_ = self.gelu(x_)
        return x_

    
class CellSpecificFinalTransform(nn.Module):
    '''Implements the final cell-specific transformation
    to chromatin accessibility prediction
    '''

    def __init__(self,
                 num_cells,
                 seq_embed_dim,
                 weight_init,
    ):
        '''
        Args:
            num_cells: output dim
            seq_embed_dim: input dim
        '''
        super(CellSpecificFinalTransform, self).__init__()

        self.seq_embed_dim = seq_embed_dim
        self.num_cells = num_cells

        self.final = nn.Linear(self.seq_embed_dim, self.num_cells)

        if weight_init == "xavier_normal":
            nn.init.xavier_normal_(self.final.weight.data)
            nn.init.constant_(self.final.bias.data, 0.01)
        elif weight_init == "xavier_uniform":
            nn.init.xavier_uniform_(self.final.weight.data)
            nn.init.constant_(self.final.bias.data, 0.01)
        elif weight_init == "kaiming_normal":
            nn.init.kaiming_normal_(self.final.weight.data)
            nn.init.constant_(self.final.bias.data, 0.01)
        elif weight_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.final.weight.data)
            nn.init.constant_(self.final.bias.data, 0.01)
        else:
            raise ValueError("Please check weight initialization method for convolution kernels")

    def forward(self, _x):
        x_ = self.final(_x)
        return torch.sigmoid(x_)

    def get_cell_embeddings(self):
        return self.final.weight.data

    def get_intercept(self):
        return self.final.bias.data


class scBasset(nn.Module):
    '''This class links StochasticReverseComplement, StochasticShift,
    SequenceEncoder and CellSpecificFinalTransform modules as scBasset model
    '''
    def __init__(self, 
                 num_cells,
                 seq_embed_dim = 32,
                 seq_shift_max = 3,
                 seq_offset = 30,
                 first_conv_out_filters=288,
                 first_conv_kernel_size=17,
                 first_conv_pool_size=3,
                 tower_conv_out_filters=512,
                 tower_conv_kernel_size=5,
                 tower_conv_pool_size=2,
                 tower_conv_repeat=6, 
                 channel_conv_out_filters=128,
                 weight_init="kaiming_normal"
    ):
        '''
        Args:
            num_cells:                  the number of cells available in all dataset
                                        splits (train, validation and test)
            seq_embed_dim:              the dimension of genome sequence embedding
            seq_shift_max:              maximum shift amount for stochastic genome
                                        sequence shifting data augmentation
            seq_offset:                 preprocessed genome sequences have `seq_offset`
                                        extra bases at each end
            first_conv_out_filters:     the number of filters of the convolution
                                        layer directly operating on genome sequence
            first_conv_kernel_size:     size of the convolution kernels
                                        operating on genome sequence
            first_conv_pool_size:       size of the pooling employed after
                                        convolution on sequence
            tower_conv_out_filters:     the number of output channels of the final
                                        convolution layer in tower
            tower_conv_kernel_size:     size of the convolution kernels
                                        in convolution tower
            tower_conv_repeat:          the number of convolution layers in the tower
            channel_conv_out_filters:   out channel size of the 1x1 convolution
                                        layer after the tower
            weight_init:                weight initialization strategy for all layers
                                        in the network
        Returns:
            [batch_size, 1] chromatin accesibility probabilities
        '''
        super(scBasset, self).__init__()

        self.num_cells = num_cells
        self.seq_embed_dim = seq_embed_dim
        self.seq_shift_max  = seq_shift_max 
        self.seq_offset  = seq_offset 
        self.first_conv_out_filters = first_conv_out_filters
        self.first_conv_kernel_size = first_conv_kernel_size
        self.first_conv_pool_size = first_conv_pool_size
        self.tower_conv_out_filters = tower_conv_out_filters
        self.tower_conv_kernel_size = tower_conv_kernel_size
        self.tower_conv_pool_size = tower_conv_pool_size
        self.tower_conv_repeat = tower_conv_repeat
        self.channel_conv_out_filters = channel_conv_out_filters
        self.weight_init = weight_init

        self.rc_augment = StochasticReverseComplement()

        self.shift_augment = StochasticShift(self.seq_shift_max, self.seq_offset)

        self.sequence_encoder = SequenceEncoder(
            seq_embed_dim = self.seq_embed_dim,
            first_conv_out_filters = self.first_conv_out_filters,
            first_conv_kernel_size = self.first_conv_kernel_size,
            first_conv_pool_size = self.first_conv_pool_size,
            tower_conv_out_filters = self.tower_conv_out_filters,
            tower_conv_kernel_size = self.tower_conv_kernel_size,
            tower_conv_pool_size = self.tower_conv_pool_size,
            tower_conv_repeat = self.tower_conv_repeat,
            channel_conv_out_filters = self.channel_conv_out_filters,
            weight_init = self.weight_init
        )

        self.final_transform =\
            CellSpecificFinalTransform(num_cells = self.num_cells,
                                       seq_embed_dim = self.seq_embed_dim,
                                       weight_init = self.weight_init
            )

    def forward(self, _x):
        x_ = self.rc_augment(_x)
        x_ = self.shift_augment(x_)
        x_ = self.sequence_encoder(x_)
        x_ = self.final_transform(x_)
        return x_

    def get_cell_embeddings(self):
        return self.final_transform.get_cell_embeddings()

    def get_intercept(self):
        return self.final_transform.get_intercept()

    def peak_embeddings(self, _x):
        '''
        Returns peak embeddings for a tensor of peak embeddings
        '''
        self.eval()
        x_ = self.rc_augment(_x)
        x_ = self.sequence_encoder(x_)
        self.train()
        return x_
