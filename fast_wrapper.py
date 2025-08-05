import math
import copy
from tqdm import tqdm
from functools import partial
import scipy
import math
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from quant import *
from utils.hw_util import *

def get_module_by_name_suffix(model, module_name: str):
	for name, module in model.named_modules():
		if name.endswith(module_name):
			return module

def _setattr(model, name, module):
	name_list = name.split(".")
	for name in name_list[:-1]:
		model = getattr(model, name)
	setattr(model, name_list[-1], module)

def device_vari_function(x, func):
	return func.pdf(x)*x**2

def cal_error(model_name, name, pim_params, sparsity, act_sparsity, var):
	wl_weight = pim_params['wl_weight']
	adc_precision = pim_params['ADCprecision']
	rram_vari = pim_params['vari']
	subArray = pim_params['subArray']
	cellbit = pim_params['cellBit']

	weight_vari = 0 # None
	device_vari = 0
	adc_vari = 0

	# device vari calculation
	adc_step_size = subArray*sparsity/2**adc_precision
	if rram_vari > 0:
		device_error_distribution = norm(loc=0, scale=math.sqrt(subArray*var*rram_vari**2))
		device_factor = scipy.integrate.quad(device_vari_function, -adc_step_size/2, adc_step_size/2, args=device_error_distribution)[0]
		device_factor /= (var*rram_vari**2)*subArray
		device_factor = max(1-device_factor,0)
		device_vari = rram_vari**2*device_factor
	
	adc_vari = ((subArray*sparsity**2/(2**(2*adc_precision)*12)))/var
	total_vari = weight_vari+device_vari+adc_vari
	weight_std = math.sqrt(total_vari)
	
	return weight_std

def wrap_error(model_name, name, weight, pim_params, index):
	sparsity, act_sparsity, var = cal_sparsity_var_factor(weight, pim_params, index)
	error = cal_error(model_name, name, pim_params, sparsity, act_sparsity, var)
	variation = torch.normal(0, math.sqrt(var)*error, weight.size()).to(weight.device)
	weight = weight + variation

	return weight


class FWConv2d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, bias=True, stride=4, padding=0, groups=None, act_quant='per_tensor', a_bit=8, dev=None):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		self.groups = groups
		
		self.a_bit = a_bit

		self.register_buffer('weight', torch.zeros(self.in_channels, 1, self.kernel_size[0], dtype=torch.float16, requires_grad=False, device=dev))
		if bias:
			self.register_buffer('bias', torch.zeros(
				(self.out_channels), dtype=torch.float16, requires_grad=False, device=dev))
		else:
			self.register_buffer('bias', None)

		if act_quant == 'per_token':
			self.act_quant_name = 'per_token'
			self.act_quant = partial(
				pseudo_qa_tensor, n_bits=self.a_bit, per_tensor=False)
		elif act_quant == 'per_tensor':
			self.act_quant_name = 'per_tensor'
			self.act_quant = partial(
				pseudo_qa_tensor, n_bits=self.a_bit, per_tensor=True)
		else:
			raise ValueError(f'Invalid act_quant: {act_quant}')

	def to(self, *args, **kwargs):
		super(FWConv2d, self).to(*args, **kwargs)
		self.weight = self.weight.to(*args, **kwargs)
		if self.bias is not None:
			self.bias = self.bias.to(*args, **kwargs)
		return self
	
	@torch.no_grad()
	def forward(self, x):
		q_x = self.act_quant(x)
		y = torch.functional.F.conv2d(q_x, self.weight, self.bias, self.stride, self.padding, 1, self.groups)
		return y

	@staticmethod
	def from_float(model_name, name, module, pim_params):
		assert isinstance(module, torch.nn.Conv2d)
		new_module = FWConv2d(module.in_channels, module.out_channels, module.kernel_size, module.bias is not None, stride=module.stride[0], padding=module.padding[0], 
							  groups=module.groups, act_quant='per_tensor', a_bit=8, dev=module.weight.device)

		weight = module.weight.reshape(module.weight.shape[0],-1)
		ideal_weight, w_scale = qw_tensor(weight, pim_params['wl_weight'], weight.shape[1], keep_shape=True)
		actual_weight = torch.zeros_like(ideal_weight)
		X_decimal = ideal_weight
		for i in range(pim_params['wl_weight']-1):
			ideal_remainder = torch.fmod(X_decimal, 2)
			actual_remainder = wrap_error(model_name, name, ideal_remainder, pim_params, i) # TODO: add index
			X_decimal = torch.round((X_decimal-ideal_remainder)/2)
			actual_weight += 2**i*actual_remainder

		new_module.weight = (actual_weight * w_scale).reshape(module.weight.shape)
		
		if module.bias is not None:
			new_module.bias = module.bias
		del module
		return new_module


class FWLinear(nn.Module):
	def __init__(self, in_features, out_features, bias=True, act_quant='per_tensor', a_bit=8, dev=None): # TODO: act quant type
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.a_bit = a_bit

		self.register_buffer('weight', torch.zeros(self.out_features,
												   self.in_features, dtype=torch.float16, requires_grad=False, device=dev))
		if bias:
			self.register_buffer('bias', torch.zeros(
				(1, self.out_features), dtype=torch.float16, requires_grad=False, device=dev))
		else:
			self.register_buffer('bias', None)

		if act_quant == 'per_token':
			self.act_quant_name = 'per_token'
			self.act_quant = partial(
				pseudo_qa_tensor, n_bits=self.a_bit, per_tensor=False)
		elif act_quant == 'per_tensor':
			self.act_quant_name = 'per_tensor'
			self.act_quant = partial(
				pseudo_qa_tensor, n_bits=self.a_bit, per_tensor=True)
		else:
			raise ValueError(f'Invalid act_quant: {act_quant}')

	def to(self, *args, **kwargs):
		super(FWLinear, self).to(*args, **kwargs)
		self.weight = self.weight.to(*args, **kwargs)
		if self.bias is not None:
			self.bias = self.bias.to(*args, **kwargs)
		return self

	@torch.no_grad()
	def forward(self, x):
		q_x = self.act_quant(x)
		y = torch.functional.F.linear(q_x, self.weight, self.bias)
		return y
	
	@staticmethod
	def from_float(model_name, name, module, pim_params):
		assert isinstance(module, torch.nn.Linear)
		new_module = FWLinear(module.in_features, module.out_features, module.bias is not None, act_quant='per_tensor', a_bit=8, dev=module.weight.device)
		ideal_weight, w_scale = qw_tensor(module.weight, pim_params['wl_weight'], module.weight.shape[1], keep_shape=True)
		actual_weight = torch.zeros_like(ideal_weight)
		X_decimal = ideal_weight
		for i in range(pim_params['wl_weight']-1):
			ideal_remainder = torch.fmod(X_decimal, 2)
			actual_remainder = wrap_error(model_name, name, ideal_remainder, pim_params, i) # TODO: add index
			X_decimal = torch.round((X_decimal-ideal_remainder)/2)
			actual_weight += 2**i*actual_remainder

		new_module.weight = (actual_weight * w_scale).reshape(module.weight.shape)
		
		if module.bias is not None:
			new_module.bias = module.bias
		del module

		return new_module


def fast_wrap_model(model_name, model, pim_params, exclude_layers=[]):
	wrapped_model = copy.deepcopy(model)
	modules_wrapper_list = []
	for name, module in wrapped_model.named_modules():
		if isinstance(module, nn.Conv2d) and module.groups == 1 and name not in exclude_layers:
			new_Conv2d = FWConv2d.from_float(model_name, name, module, pim_params)
			father_module = get_module_by_name_suffix(wrapped_model, '.'.join(name.split('.')[:-1]))
			setattr(father_module, name.split('.')[-1], new_Conv2d)
			del new_Conv2d, module
			torch.cuda.empty_cache()
		elif isinstance(module, nn.Linear) and name not in exclude_layers:
			new_Linear = FWLinear.from_float(model_name, name, module, pim_params)
			father_module = get_module_by_name_suffix(wrapped_model, '.'.join(name.split('.')[:-1]))
			setattr(father_module, name.split('.')[-1], new_Linear)
			del new_Linear, module
			torch.cuda.empty_cache()
	
	return wrapped_model


if __name__ == "__main__":
	pim_params={}
	pim_params['wl_weight'] = 8
	pim_params['wl_input'] = 8
	# Hardware Properties
	pim_params['subArray'] = 256           # size of subArray (e.g. 128*128)
	pim_params['ADCprecision'] = 8         # ADC precision (e.g. 5-bit)
	pim_params['cellBit'] = 1              # cell precision (e.g. 4-bit/cell)
	pim_params['onoffratio'] = 1e6         # device on/off ratio (e.g. Gmax/Gmin = 3)
	# if do not run the device retention / conductance variation effects, set args.vari=0, args.v=0
	pim_params['vari'] = 0                 # conductance variation (e.g. 0.1 standard deviation to generate random variation)