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
import scipy
import math
from scipy.stats import norm

from wrapper import *
from util import get_cnn_model, get_llm_model, get_dataset, replace_module, AverageMeter

home_path = os.getcwd()
SimConfig_path = os.path.join(home_path, 'Search_space.ini')
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="path of the hf model")
parser.add_argument("--dataset", type=str, default='imagenet', help="the evaluation dataset")
parser.add_argument("--path", type=str, help="save path")
parser.add_argument("--specific_layer", type=str, default=None, help="save path")
parser.add_argument('--gpu_id', type=int, default=None, help='GPU id')
parser.add_argument('--cycle', type=int, default=3, help='the cycle number of one sample')
parser.add_argument('--device_only', action='store_true', help='whether to model device error independently')
params = parser.parse_args()

def device_vari_function(x, func):
	return func.pdf(x)*x**2

def device_vari_function_test(x, func):
	return func.pdf(x)

def calculate_variance(prop, var_w, abs_mean_w, var_d):
	total_variance = prop * var_w + prop * var_w + (1-2*prop)*var_d
	
	return total_variance

def cal_qcn(model_name, pim_params, device_only=False):
	wl_weight = pim_params['wl_weight']
	adc_precision = pim_params['ADCprecision']
	vari_type = pim_params['vari_type']
	if vari_type == 'state_dependent':
		rram_vari = pim_params['vari']
	elif pim_params['vari_type'] == 'state_independent':
		rram_ind_vari = pim_params['ind_vari']
	elif pim_params['vari_type'] == 'laplace':
		laplace_b = pim_params['laplace_b']

	SAF_prop = pim_params['SAF_prop']
	subArray = pim_params['subArray']
	cellbit = pim_params['cellBit']

	if params.specific_layer is not None:
		sparsity = np.loadtxt('./prior_stat/{}/{}/{}b_{}size_sparsity.txt'.format(model_name, params.specific_layer, wl_weight, subArray))
		act_sparsity = np.loadtxt('./prior_stat/{}/{}/{}b_{}size_act_sparsity.txt'.format(model_name, params.specific_layer, wl_weight, subArray))
		var = np.loadtxt('./prior_stat/{}/{}/{}b_{}size_var.txt'.format(model_name, params.specific_layer, wl_weight, subArray))
		abs_mean = np.loadtxt('./prior_stat/{}/{}/{}b_{}size_abs_mean.txt'.format(model_name, params.specific_layer, wl_weight, subArray))
		base_var = np.loadtxt('./prior_stat/{}/{}/base_var.txt'.format(model_name, params.specific_layer, )).mean()
	else:
		sparsity = np.loadtxt('./prior_stat/{}/{}b_{}size_sparsity.txt'.format(model_name, wl_weight, subArray))
		act_sparsity = np.loadtxt('./prior_stat/{}/{}b_{}size_act_sparsity.txt'.format(model_name, wl_weight, subArray))
		var = np.loadtxt('./prior_stat/{}/{}b_{}size_var.txt'.format(model_name, wl_weight, subArray))
		abs_mean = np.loadtxt('./prior_stat/{}/{}b_{}size_abs_mean.txt'.format(model_name, wl_weight, subArray))
		base_var = np.loadtxt('./prior_stat/{}/base_var.txt'.format(model_name)).mean()

	total_qcn = 0
	ideal = 0
	weight_error = 0
	device_error = 0
	adc_error = 0

	for i in range(wl_weight-1):
		ideal += 2**(2*i)*var[i]
		adc_step_size = subArray*sparsity[i]/2**adc_precision
		if vari_type == 'state_dependent':
			vari = calculate_variance(SAF_prop,var[i],abs_mean[i],var[i]*rram_vari**2)
			if vari > 0:
				device_error_distribution = norm(loc=0, scale=math.sqrt(subArray*vari))
				device_factor = scipy.integrate.quad(device_vari_function, -adc_step_size/2, adc_step_size/2, args=device_error_distribution)[0]
				device_factor /= vari*subArray
				# device_factor = scipy.integrate.quad(device_vari_function_test, -adc_step_size/2, adc_step_size/2, args=device_error_distribution)[0]
				# import ipdb; ipdb.set_trace()
				device_factor = max(1-device_factor,0)
			else:
				device_factor = 1
			# device_error += 2**(2*i)*(var[i]*rram_vari**2)*device_factor
			device_error += 2**(2*i)*vari*device_factor
		elif vari_type == 'laplace':
			vari = calculate_variance(SAF_prop,var[i],abs_mean[i],var[i]*2*laplace_b**2)
			if vari > 0:
				device_error_distribution = norm(loc=0, scale=math.sqrt(subArray*vari))
				device_factor = scipy.integrate.quad(device_vari_function, -adc_step_size/2, adc_step_size/2, args=device_error_distribution)[0]
				device_factor /= vari*subArray
				device_factor = max(1-device_factor,0)
			else:
				device_factor = 1
			# device_error += 2**(2*i)*(var[i]*rram_vari**2)*device_factor
			device_error += 2**(2*i)*vari*device_factor
		else:
			vari = calculate_variance(SAF_prop,var[i],abs_mean[i],rram_ind_vari**2)
			if vari > 0:
				device_error_distribution = norm(loc=0, scale=math.sqrt(subArray*vari))
				device_factor = scipy.integrate.quad(device_vari_function, -adc_step_size/2, adc_step_size/2, args=device_error_distribution)[0]
				device_factor /= vari*subArray
				device_factor = max(1-device_factor,0)
			else:
				device_factor = 1
			device_error += 2**(2*i)*vari*device_factor
		adc_error += 2**(2*i)*((subArray*sparsity[i]**2/(2**(2*adc_precision)*12)))/math.sqrt(act_sparsity[i])
		# import ipdb; ipdb.set_trace()
		# adc_error += 2**(2*i)*((subArray*sparsity[i]**2/(2**(2*adc_precision)*12)))

	weight_error = base_var*3/(2**(2*wl_weight))
	base_scale = 6*math.sqrt(base_var)/2**wl_weight
	weight_error = weight_error/(base_scale**2)
	# import ipdb; ipdb.set_trace()
	# weight_error = 3*ideal/(2**(2*wl_weight))

	if device_only:
		error = device_error+1e-2
	else:
		error = weight_error + device_error + adc_error
	total_qcn = ideal/error
	return total_qcn


def main():
	# os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_id)
	device = torch.device("cuda")

	acc_list = []
	pim_params_list = []
	if 'mamba' not in params.model_name and 'rwkv' not in params.model_name and 'opt' not in params.model_name:
		if params.device_only:
			f = open("/share/guolidong-nfs/PIM/PA-SIM/results/device/{}/{}_accsim.log".format(params.model_name, params.dataset),"r")
		else:
			f = open("/share/guolidong-nfs/PIM/PA-SIM/results/{}/{}_accsim.log".format(params.model_name, params.dataset),"r")
		lines = f.readlines()
		for line in lines:
			if 'wl_weight' in line:
				pim_params = {}
				pim_params['wl_weight'] = int(line.split("'wl_weight': ")[1][0])
				pim_params['ADCprecision'] = int(line.split("'ADCprecision': ")[1][0])
				pim_params['vari_type'] = str(line.split("'vari_type': ")[1].split(',')[0][1:-1])
				if pim_params['vari_type'] == 'state_dependent':
					pim_params['vari'] = float(line.split("'vari': ")[1].split(',')[0])
				elif pim_params['vari_type'] == 'state_independent':
					pim_params['ind_vari'] = float(line.split("'ind_vari': ")[1].split(',')[0])
				elif pim_params['vari_type'] == 'laplace':
					pim_params['laplace_b'] = float(line.split("'laplace_b': ")[1].split(',')[0])
				pim_params['SAF_prop'] = float(line.split("'SAF_prop': ")[1].split(',')[0][:-2])
				pim_params['subArray'] = int(line.split("'subArray': ")[1].split(',')[0])
				pim_params['cellBit'] = int(line.split("'cellBit': ")[1][0])
				pim_params_list.append(pim_params)
				# if pim_params not in pim_params_list:
				# 	pim_params_list.append(pim_params)
				# 	acc_repeat = False
				# else:
				# 	acc_repeat=True
			if 'Acc' in line:
				acc = float(line.split('=')[-1][:-1])
				acc_list.append(acc)
	else:
		if params.specific_layer:
			f = open("/share/guolidong-nfs/PIM/PA-SIM/results/{}/{}_{}_accsim.log".format(params.model_name, params.specific_layer, params.dataset),"r")
		else:
			f = open("/share/guolidong-nfs/PIM/PA-SIM/results/{}/{}_accsim.log".format(params.model_name, params.dataset),"r")
		lines = f.readlines()
		for line in lines:
			if 'acc,none' in line:
				acc = float(line.split('acc,none')[1].split(',')[0][3:])
				acc_list.append(acc)
			if 'wl_weight' in line:
				pim_params = {}
				pim_params['wl_weight'] = int(line.split("'wl_weight': ")[1][0])
				pim_params['ADCprecision'] = int(line.split("'ADCprecision': ")[1][0])
				pim_params['vari_type'] = str(line.split("'vari_type': ")[1].split(',')[0][1:-1])
				if pim_params['vari_type'] == 'state_dependent':
					pim_params['vari'] = float(line.split("'vari': ")[1].split(',')[0])
				elif pim_params['vari_type'] == 'state_independent':
					pim_params['ind_vari'] = float(line.split("'ind_vari': ")[1].split(',')[0])
				elif pim_params['vari_type'] == 'laplace':
					pim_params['laplace_b'] = float(line.split("'laplace_b': ")[1].split(',')[0])
				pim_params['SAF_prop'] = float(line.split("'SAF_prop': ")[1].split(',')[0][:-2])
				pim_params['subArray'] = int(line.split("'subArray': ")[1].split(',')[0])
				pim_params['cellBit'] = int(line.split("'cellBit': ")[1][0])
				if pim_params not in pim_params_list:
					pim_params_list.append(pim_params)
	
	qcn_list = []
	for pim_params in pim_params_list:
		qcn = cal_qcn(params.model_name, pim_params, device_only=params.device_only)
		qcn_list.append(qcn)

	_acc_list = np.array(acc_list).reshape(-1,params.cycle).mean(-1).tolist()
	acc_list = np.array(acc_list).reshape(-1,params.cycle)

	remove_index = []
	for i in range(len(_acc_list)):
		# i_adc_precision = pim_params_list[i]['ADCprecision']
		if pim_params_list[i]['SAF_prop']!=1e-3:
			remove_index.append(i)

	remove_index = set(remove_index)
	# remove_index = []

	# delete 2 cell bits
	filtered_qcn_list = []
	filtered_acc_list = []
	filtered_all_acc_list = []
	for i in range(len(_acc_list)):
		if pim_params_list[i]['cellBit']==1 and i not in remove_index:
			filtered_acc_list.append(_acc_list[i])
			filtered_qcn_list.append(qcn_list[i])
			filtered_all_acc_list.append(acc_list[i][None,:])
			# if math.log(qcn_list[i],2)<-2:
			# 	print(pim_params_list[i])

	kendalltau = scipy.stats.kendalltau(np.array(filtered_qcn_list), np.array(filtered_acc_list))
	print("kendalltau is {}".format(kendalltau))

	# import ipdb; ipdb.set_trace()
	# save the results to txt for origin picture
	final_results = np.concatenate((np.array(filtered_qcn_list)[:,None], np.concatenate(filtered_all_acc_list)),1)
	np.savetxt(os.path.join(params.path, '2_{}_nipa_results.txt'.format(params.dataset)), final_results)
	# np.savetxt('./results/{}/{}_nipa_results.txt'.format(params.model_name, params.dataset), final_results)

	font=16
	size=(8, 8)
	fig = plt.figure(figsize=size)
	plt.scatter(np.log2(np.array(filtered_qcn_list)), np.array(filtered_acc_list), s=50, marker='.')
	plt.xticks(fontsize=font)
	plt.yticks(fontsize=font)
	plt.xlabel('$NIPA-Metric$', fontsize=font)
	plt.ylabel('$Accuracy$', fontsize=font)
	plt.grid(True)
	fig.tight_layout()
	fig.savefig(os.path.join(params.path, '2_{}_nipa_fitting.png'.format(params.dataset)), dpi=600)
	# fig.savefig('./results/{}/{}_nipa_fitting.png'.format(params.model_name, params.dataset), dpi=600)

if __name__ == "__main__":
	main()