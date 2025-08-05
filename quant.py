import os
import torch
import numpy as np
import copy
import math

@torch.no_grad()
def qw_tensor(tensor, n_bits=8, q_group_size=-1, per_tensor=False, keep_shape=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """

    org_tensor_shape = tensor.shape
    if q_group_size > 0:
        tensor = tensor.reshape(-1, q_group_size)
    if len(org_tensor_shape)==4:
        # per out channel in Conv2d
        tensor = tensor.reshape(tensor.shape[0],-1)
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2

    max_val = tensor.abs().amax(dim=1, keepdim=True)
    max_val = max_val.clamp(min=1e-5)
    max_int = 2 ** (n_bits - 1) - 1
    min_int = -(2 ** (n_bits - 1)) + 1
    scales = max_val / max_int

    tensor = torch.clamp(torch.round(tensor / scales), min_int, max_int)

    # (tensor.div_(scales).round_()).clamp_(min_int, max_int)

    assert torch.isnan(tensor).sum() == 0

    if keep_shape is False:
        tensor = tensor.reshape(org_tensor_shape)

    return tensor, scales

@torch.no_grad()
def qa_tensor(tensor, n_bits=8, q_group_size=-1, per_tensor=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    org_tensor_shape = tensor.shape
    if q_group_size > 0:
        tensor = tensor.reshape(-1, q_group_size)
    if len(org_tensor_shape)==4:
        # per out channel in Conv2d
        tensor = tensor.reshape(tensor.shape[0],-1)
    elif len(org_tensor_shape)==3:
        # llm
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if per_tensor:
        tensor = tensor.reshape(1, -1)
    # assert tensor.dim() == 2

    max_val = tensor.abs().amax(dim=1, keepdim=True)
    max_val = max_val.clamp(min=1e-5)
    max_int = 2 ** (n_bits - 1) - 1
    min_int = -(2 ** (n_bits - 1)) + 1
    scales = max_val / max_int

    tensor = torch.clamp(torch.round(tensor / scales), min_int, max_int)

    assert torch.isnan(tensor).sum() == 0

    tensor = tensor.reshape(org_tensor_shape)

    return tensor, scales

@torch.no_grad()
def pseudo_qa_tensor(tensor, n_bits=8, q_group_size=-1, per_tensor=False):
    """
    The basic quantization function for weight, activation and KV cache.
    """
    org_tensor_shape = tensor.shape
    if q_group_size > 0:
        tensor = tensor.reshape(-1, q_group_size)
    if len(org_tensor_shape)==4:
        # per out channel in Conv2d
        tensor = tensor.reshape(tensor.shape[0],-1)
    elif len(org_tensor_shape)==3:
        # llm
        tensor = tensor.reshape(-1, tensor.shape[-1])
    if per_tensor:
        tensor = tensor.reshape(1, -1)

    max_val = tensor.abs().amax(dim=1, keepdim=True)
    max_val = max_val.clamp(min=1e-5)
    max_int = 2 ** (n_bits - 1) - 1
    min_int = -(2 ** (n_bits - 1)) + 1
    scales = max_val / max_int

    tensor = torch.clamp(torch.round(tensor / scales), min_int, max_int) * scales

    assert torch.isnan(tensor).sum() == 0

    tensor = tensor.reshape(org_tensor_shape)

    return tensor


def LinearQuantizeOut(x, bit):
    # return x
    if torch.max(x.abs()) == 0:
        y = x
    else:
        minQ = torch.min(x)
        delta = torch.max(x) - torch.min(x)
        y = x.clone()

        stepSizeRatio = 2.**(-bit)
        stepSize = stepSizeRatio*delta.item()
        index = torch.clamp(torch.round((x-minQ.item())/stepSize), 0, (2.**(bit)))
        y = index*stepSize + minQ.item()

    return y