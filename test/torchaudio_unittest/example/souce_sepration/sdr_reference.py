"""Reference Implementation of SDR and PIT SDR.

This module was taken from the following implementation

https://github.com/naplab/Conv-TasNet/blob/e66d82a8f956a69749ec8a4ae382217faa097c5c/utility/sdr.py

which was made available by Yi Luo under the following liscence,

Creative Commons Attribution-NonCommercial-ShareAlike 3.0 United States License.

The module was modified in the following manner;
 - Remove the functions other than `calc_sdr_torch` and `batch_SDR_torch`,
 - Remove the import statements required only for the removed functions.
 - Add `# flake8: noqa` so as not to report any format issue on this module.

The implementation of the retained functions and their formats are kept as-is.
"""

# flake8: noqa

import numpy as np
from itertools import permutations

import torch


def calc_sdr_torch(estimation, origin, mask=None):
    """
    batch-wise SDR caculation for one audio file on pytorch Variables.
    estimation: (batch, nsample)
    origin: (batch, nsample)
    mask: optional, (batch, nsample), binary
    """
    
    if mask is not None:
        origin = origin * mask
        estimation = estimation * mask
    
    origin_power = torch.pow(origin, 2).sum(1, keepdim=True) + 1e-8  # (batch, 1)
    
    scale = torch.sum(origin*estimation, 1, keepdim=True) / origin_power  # (batch, 1)
    
    est_true = scale * origin  # (batch, nsample)
    est_res = estimation - est_true  # (batch, nsample)
    
    true_power = torch.pow(est_true, 2).sum(1)
    res_power = torch.pow(est_res, 2).sum(1)
    
    return 10*torch.log10(true_power) - 10*torch.log10(res_power)  # (batch, 1)


def batch_SDR_torch(estimation, origin, mask=None):
    """
    batch-wise SDR caculation for multiple audio files.
    estimation: (batch, nsource, nsample)
    origin: (batch, nsource, nsample)
    mask: optional, (batch, nsample), binary
    """
    
    batch_size_est, nsource_est, nsample_est = estimation.size()
    batch_size_ori, nsource_ori, nsample_ori = origin.size()
    
    assert batch_size_est == batch_size_ori, "Estimation and original sources should have same shape."
    assert nsource_est == nsource_ori, "Estimation and original sources should have same shape."
    assert nsample_est == nsample_ori, "Estimation and original sources should have same shape."
    
    assert nsource_est < nsample_est, "Axis 1 should be the number of sources, and axis 2 should be the signal."
    
    batch_size = batch_size_est
    nsource = nsource_est
    nsample = nsample_est
    
    # zero mean signals
    estimation = estimation - torch.mean(estimation, 2, keepdim=True).expand_as(estimation)
    origin = origin - torch.mean(origin, 2, keepdim=True).expand_as(estimation)
    
    # possible permutations
    perm = list(set(permutations(np.arange(nsource))))
    
    # pair-wise SDR
    SDR = torch.zeros((batch_size, nsource, nsource)).type(estimation.type())
    for i in range(nsource):
        for j in range(nsource):
            SDR[:,i,j] = calc_sdr_torch(estimation[:,i], origin[:,j], mask)
    
    # choose the best permutation
    SDR_max = []
    SDR_perm = []
    for permute in perm:
        sdr = []
        for idx in range(len(permute)):
            sdr.append(SDR[:,idx,permute[idx]].view(batch_size,-1))
        sdr = torch.sum(torch.cat(sdr, 1), 1)
        SDR_perm.append(sdr.view(batch_size, 1))
    SDR_perm = torch.cat(SDR_perm, 1)
    SDR_max, _ = torch.max(SDR_perm, dim=1)
    
    return SDR_max / nsource
