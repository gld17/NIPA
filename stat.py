import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import random
import logging
import configparser as cp
import sys
import torchvision.models as models
from torchvision import transforms
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, RwkvForCausalLM, MambaForCausalLM
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator, tasks
from lm_eval.tasks import get_task_dict

from wrapper import *
from util import get_cnn_model, get_llm_model, get_dataset, replace_module

home_path = os.getcwd()
SimConfig_path = os.path.join(home_path, 'Search_space.ini')
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="path of the hf model")
parser.add_argument('--gpu_id', type=int, default=None, help='GPU id')
parser.add_argument('--wl_weight', type=int, default=8, help='the bit-width of weights')
parser.add_argument('--subArray', type=int, default=256, help='the size of xbar')
parser.add_argument('--cellbit', type=int, default=1, help='the resolution of RRAM device')
params = parser.parse_args()

def main():
	# os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_id)
	device = torch.device("cuda")
	
	if 'mamba' in params.model_name or 'rwkv' in params.model_name or 'opt' in params.model_name:
		if 'mamba' in params.model_name:
			model = MambaForCausalLM.from_pretrained(params.model_name, device_map=device, torch_dtype=torch.float16)
			enc = AutoTokenizer.from_pretrained(params.model_name)
			
			exclude_layers = []
			non_exclude_layers = []
			for name, module in model.named_modules():
				if 'layers.0' not in name or 'dt_proj' in name:
					exclude_layers.append(name)
				else:
					non_exclude_layers.append(name)
		elif 'rwkv' in params.model_name:
			model = RwkvForCausalLM.from_pretrained(params.model_name, device_map=device, torch_dtype=torch.float16)
			enc = AutoTokenizer.from_pretrained(params.model_name)

			exclude_layers = []
			non_exclude_layers = []
			for name, module in model.named_modules():
				if 'blocks.0' not in name:
					exclude_layers.append(name)
				else:
					non_exclude_layers.append(name)
		elif 'opt' in params.model_name:
			model = AutoModelForCausalLM.from_pretrained(params.model_name, device_map=device).half()
			enc = AutoTokenizer.from_pretrained(params.model_name)
			
			exclude_layers = []
			non_exclude_layers = []
			for name, module in model.named_modules():
				if 'layers.23' not in name:
					exclude_layers.append(name)
				else:
					non_exclude_layers.append(name)
		model_name = params.model_name.split('/')[-1]
	else:
		model, exclude_layers = get_cnn_model(params.model_name)
		model = model.to(device).half()
		model.eval()
		model_name = params.model_name
	subArray = params.subArray
	cellbit = params.cellbit
	cellrange = 2**cellbit
	wl_weight = params.wl_weight

	non_zero_ratios_0 = []
	non_zero_ratios_1 = []
	non_zero_ratios_2 = []
	non_zero_ratios_3 = []
	non_zero_ratios_4 = []
	non_zero_ratios_5 = []
	non_zero_ratios_6 = []
	act_non_zero_ratios_0 = []
	act_non_zero_ratios_1 = []
	act_non_zero_ratios_2 = []
	act_non_zero_ratios_3 = []
	act_non_zero_ratios_4 = []
	act_non_zero_ratios_5 = []
	act_non_zero_ratios_6 = []
	var_0 = []
	var_1 = []
	var_2 = []
	var_3 = []
	var_4 = []
	var_5 = []
	var_6 = []
	abs_mean_0 = []
	abs_mean_1 = []
	abs_mean_2 = []
	abs_mean_3 = []
	abs_mean_4 = []
	abs_mean_5 = []
	abs_mean_6 = []
	factors = []
	base_var = []
	for name, module in model.named_modules():
		if isinstance(module, nn.Linear) and name not in exclude_layers:
			weight = module.weight
		elif isinstance(module, nn.Conv2d) and name not in exclude_layers and module.groups==1:
			weight = module.weight.reshape(module.weight.shape[0],-1)
		else:
			continue
		
		factor = weight.shape[1]/subArray
		factors.append(max(factor,1))
		base_var.append(torch.var(weight).item())
		
		if weight.shape[1] <= subArray:
			padding_num = subArray-weight.shape[1]
			weight = F.pad(weight,(0,padding_num),'constant',0)
			weight, w_scale = qw_tensor(weight, wl_weight, subArray)
			X_decimal = weight
			for k in range(wl_weight-1):
				remainder = torch.fmod(X_decimal, cellrange)
				X_decimal = torch.round((X_decimal-remainder)/cellrange)
				ratio = ((remainder==1).sum(-1).float().mean()+(remainder==-1).sum(-1).float().mean())/(remainder>-2).sum(-1)[0]
				act_ratio = remainder.abs().sum(-1).mean()/(remainder>-2).sum(-1)[0]
				if k==0:
					var_0.append(torch.var(remainder,1).mean().item())
					abs_mean_0.append(remainder.abs().mean().item())
					non_zero_ratios_0.append(ratio.item()/math.sqrt(factor))
					act_non_zero_ratios_0.append(act_ratio.item())
				elif k==1:
					var_1.append(torch.var(remainder,1).mean().item())
					abs_mean_1.append(remainder.abs().mean().item())
					non_zero_ratios_1.append(ratio.item()/math.sqrt(factor))
					act_non_zero_ratios_1.append(act_ratio.item())
				elif k==2:
					var_2.append(torch.var(remainder,1).mean().item())
					abs_mean_2.append(remainder.abs().mean().item())
					non_zero_ratios_2.append(ratio.item()/math.sqrt(factor))
					act_non_zero_ratios_2.append(act_ratio.item())
				elif k==3:
					var_3.append(torch.var(remainder,1).mean().item())
					abs_mean_3.append(remainder.abs().mean().item())
					non_zero_ratios_3.append(ratio.item()/math.sqrt(factor))
					act_non_zero_ratios_3.append(act_ratio.item())
				elif k==4:
					var_4.append(torch.var(remainder,1).mean().item())
					abs_mean_4.append(remainder.abs().mean().item())
					non_zero_ratios_4.append(ratio.item()/math.sqrt(factor))
					act_non_zero_ratios_4.append(act_ratio.item())
				elif k==5:
					var_5.append(torch.var(remainder,1).mean().item())
					abs_mean_5.append(remainder.abs().mean().item())
					non_zero_ratios_5.append(ratio.item()/math.sqrt(factor))
					act_non_zero_ratios_5.append(act_ratio.item())
				elif k==6:
					var_6.append(torch.var(remainder,1).mean().item())
					abs_mean_6.append(remainder.abs().mean().item())
					non_zero_ratios_6.append(ratio.item()/math.sqrt(factor))
					act_non_zero_ratios_6.append(act_ratio.item())
		else:
			numSubArray = math.ceil(weight.shape[1]/subArray)
			padding_num = numSubArray*subArray-weight.shape[1]
			weight = F.pad(weight,(0,padding_num),'constant',0)

			weight = weight[:,:numSubArray*subArray]
			weight, w_scale = qw_tensor(weight, wl_weight, subArray)
			weight = weight.reshape(-1, numSubArray, subArray)
			_non_zero_ratios_0 = []
			_non_zero_ratios_1 = []
			_non_zero_ratios_2 = []
			_non_zero_ratios_3 = []
			_non_zero_ratios_4 = []
			_non_zero_ratios_5 = []
			_non_zero_ratios_6 = []
			_act_non_zero_ratios_0 = []
			_act_non_zero_ratios_1 = []
			_act_non_zero_ratios_2 = []
			_act_non_zero_ratios_3 = []
			_act_non_zero_ratios_4 = []
			_act_non_zero_ratios_5 = []
			_act_non_zero_ratios_6 = []
			for xbar_index in range(numSubArray):
				X_decimal = weight[:, xbar_index]
				for k in range(wl_weight-1):
					remainder = torch.fmod(X_decimal, cellrange)
					X_decimal = torch.round((X_decimal-remainder)/cellrange)
					ratio = ((remainder==1).sum(-1).float().mean()+(remainder==-1).sum(-1).float().mean())/(remainder>-2).sum(-1)[0]
					act_ratio = remainder.abs().sum(-1).mean()/(remainder>-2).sum(-1)[0]
					if k==0:
						var_0.append(torch.var(remainder,1).mean().item())
						abs_mean_0.append(remainder.abs().mean().item())
						_non_zero_ratios_0.append(ratio.item()**2)
						_act_non_zero_ratios_0.append(act_ratio.item())
					elif k==1:
						var_1.append(torch.var(remainder,1).mean().item())
						abs_mean_1.append(remainder.abs().mean().item())
						_non_zero_ratios_1.append(ratio.item()**2)
						_act_non_zero_ratios_1.append(act_ratio.item())
					elif k==2:
						var_2.append(torch.var(remainder,1).mean().item())
						abs_mean_2.append(remainder.abs().mean().item())
						_non_zero_ratios_2.append(ratio.item()**2)
						_act_non_zero_ratios_2.append(act_ratio.item())
					elif k==3:
						var_3.append(torch.var(remainder,1).mean().item())
						abs_mean_3.append(remainder.abs().mean().item())
						_non_zero_ratios_3.append(ratio.item()**2)
						_act_non_zero_ratios_3.append(act_ratio.item())
					elif k==4:
						var_4.append(torch.var(remainder,1).mean().item())
						abs_mean_4.append(remainder.abs().mean().item())
						_non_zero_ratios_4.append(ratio.item()**2)
						_act_non_zero_ratios_4.append(act_ratio.item())
					elif k==5:
						var_5.append(torch.var(remainder,1).mean().item())
						abs_mean_5.append(remainder.abs().mean().item())
						_non_zero_ratios_5.append(ratio.item()**2)
						_act_non_zero_ratios_5.append(act_ratio.item())
					elif k==6:
						var_6.append(torch.var(remainder,1).mean().item())
						abs_mean_6.append(remainder.abs().mean().item())
						_non_zero_ratios_6.append(ratio.item()**2)
						_act_non_zero_ratios_6.append(act_ratio.item())

			# if len(_non_zero_ratios_0)>0:					
			# 	non_zero_ratios_0.append(math.sqrt(np.mean(_non_zero_ratios_0)*numSubArray))
			# if len(_non_zero_ratios_1)>0:
			# 	non_zero_ratios_1.append(math.sqrt(np.mean(_non_zero_ratios_1)*numSubArray))
			# if len(_non_zero_ratios_2)>0:
			# 	non_zero_ratios_2.append(math.sqrt(np.mean(_non_zero_ratios_2)*numSubArray))
			# if len(_non_zero_ratios_3)>0:
			# 	non_zero_ratios_3.append(math.sqrt(np.mean(_non_zero_ratios_3)*numSubArray))
			# if len(_non_zero_ratios_4)>0:
			# 	non_zero_ratios_4.append(math.sqrt(np.mean(_non_zero_ratios_4)*numSubArray))
			# if len(_non_zero_ratios_5)>0:
			# 	non_zero_ratios_5.append(math.sqrt(np.mean(_non_zero_ratios_5)*numSubArray))
			# if len(_non_zero_ratios_6)>0:
			# 	non_zero_ratios_6.append(math.sqrt(np.mean(_non_zero_ratios_6)*numSubArray))
			
			if len(_non_zero_ratios_0)>0:					
				non_zero_ratios_0.append(math.sqrt(np.mean(_non_zero_ratios_0)*numSubArray**2/factor))
				act_non_zero_ratios_0.append(np.mean(_act_non_zero_ratios_0))
			if len(_non_zero_ratios_1)>0:
				non_zero_ratios_1.append(math.sqrt(np.mean(_non_zero_ratios_1)*numSubArray**2/factor))
				act_non_zero_ratios_1.append(np.mean(_act_non_zero_ratios_1))
			if len(_non_zero_ratios_2)>0:
				non_zero_ratios_2.append(math.sqrt(np.mean(_non_zero_ratios_2)*numSubArray**2/factor))
				act_non_zero_ratios_2.append(np.mean(_act_non_zero_ratios_2))
			if len(_non_zero_ratios_3)>0:
				non_zero_ratios_3.append(math.sqrt(np.mean(_non_zero_ratios_3)*numSubArray**2/factor))
				act_non_zero_ratios_3.append(np.mean(_act_non_zero_ratios_3))
			if len(_non_zero_ratios_4)>0:
				non_zero_ratios_4.append(math.sqrt(np.mean(_non_zero_ratios_4)*numSubArray**2/factor))
				act_non_zero_ratios_4.append(np.mean(_act_non_zero_ratios_4))
			if len(_non_zero_ratios_5)>0:
				non_zero_ratios_5.append(math.sqrt(np.mean(_non_zero_ratios_5)*numSubArray**2/factor))
				act_non_zero_ratios_5.append(np.mean(_act_non_zero_ratios_5))
			if len(_non_zero_ratios_6)>0:
				non_zero_ratios_6.append(math.sqrt(np.mean(_non_zero_ratios_6)*numSubArray**2/factor))
				act_non_zero_ratios_6.append(np.mean(_act_non_zero_ratios_6))
			
	sparsity_list = []
	act_sparsity_list = []
	var_list = []
	abs_mean_list = []
	for i in range(wl_weight-1):
		if i==0:
			sparsity_list.append(np.mean(non_zero_ratios_0))
			act_sparsity_list.append(np.mean(act_non_zero_ratios_0))
			var_list.append(np.mean(var_0))
			abs_mean_list.append(np.mean(abs_mean_0))
		elif i==1:
			sparsity_list.append(np.mean(non_zero_ratios_1))
			act_sparsity_list.append(np.mean(act_non_zero_ratios_1))
			var_list.append(np.mean(var_1))
			abs_mean_list.append(np.mean(abs_mean_1))
		elif i==2:
			sparsity_list.append(np.mean(non_zero_ratios_2))
			act_sparsity_list.append(np.mean(act_non_zero_ratios_2))
			var_list.append(np.mean(var_2))
			abs_mean_list.append(np.mean(abs_mean_2))
		elif i==3:
			sparsity_list.append(np.mean(non_zero_ratios_3))
			act_sparsity_list.append(np.mean(act_non_zero_ratios_3))
			var_list.append(np.mean(var_3))
			abs_mean_list.append(np.mean(abs_mean_3))
		elif i==4:
			sparsity_list.append(np.mean(non_zero_ratios_4))
			act_sparsity_list.append(np.mean(act_non_zero_ratios_4))
			var_list.append(np.mean(var_4))
			abs_mean_list.append(np.mean(abs_mean_4))
		elif i==5:
			sparsity_list.append(np.mean(non_zero_ratios_5))
			act_sparsity_list.append(np.mean(act_non_zero_ratios_5))
			var_list.append(np.mean(var_5))
			abs_mean_list.append(np.mean(abs_mean_5))
		elif i==6:
			sparsity_list.append(np.mean(non_zero_ratios_6))
			act_sparsity_list.append(np.mean(act_non_zero_ratios_6))
			var_list.append(np.mean(var_6))
			abs_mean_list.append(np.mean(abs_mean_6))

	np.savetxt('./prior_stat/{}/base_var.txt'.format(model_name), base_var)
	np.savetxt('./prior_stat/{}/{}b_{}size_act_sparsity.txt'.format(model_name, params.wl_weight, params.subArray), act_sparsity_list)
	np.savetxt('./prior_stat/{}/{}b_{}size_sparsity.txt'.format(model_name, params.wl_weight, params.subArray), sparsity_list)
	np.savetxt('./prior_stat/{}/{}b_{}size_var.txt'.format(model_name, params.wl_weight, params.subArray), var_list)
	np.savetxt('./prior_stat/{}/{}b_{}size_abs_mean.txt'.format(model_name, params.wl_weight, params.subArray), abs_mean_list)

if __name__ == "__main__":
	main()