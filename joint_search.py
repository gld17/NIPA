#!/usr/bin/python
# -*-coding:utf-8-*-
import os
import sys
import math
import argparse
import numpy as np
import torch
import collections
import configparser
import time
import configparser as cp
import copy
from collections import Counter
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import torchvision.models as models
from torchvision import transforms
import torchvision
from torch.autograd import Variable
import scipy
from scipy.stats import norm

from wrapper import *
from utils.hw_util import *
from util import get_cnn_model, get_dataset, replace_module

home_path = os.getcwd()
SimConfig_path = os.path.join(home_path, 'HW_config', 'Joint_Search_Space.ini')
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="path of the hf model")
parser.add_argument('--gpu_id', type=str, default=None, help='GPU id')
parser.add_argument('--sample_num', type=int, default=200, help='the number of samples')
parser.add_argument("--HWdes", "--hardware_description", default=SimConfig_path)
parser.add_argument('--log_file', type=str, default='./test.log', help='log file')
params = parser.parse_args()

LOG = logging.getLogger('main')
LOG.setLevel(logging.INFO)
fh = logging.FileHandler(params.log_file)
LOG_FORMAT = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(LOG_FORMAT)
LOG.addHandler(fh)

from Hardware_Model.RRAM_PE import ReRAM_ProcessElement
from Hardware_Model.BUF import Buffer
from Hardware_Model.LUT import Look_Up_Table

model_to_dimension = {
	"mamba-130m": 768,
	"mamba-370m": 1024,
	"mamba-1.4b": 2048,
	"mamba-2.8b": 2536,
	"rwkv-169m": 768,
	"rwkv-430m": 1024,
	"rwkv-1.5b": 2048,
	"rwkv-3b": 2536,
}

def device_vari_function(x, func):
	return func.pdf(x)*x**2

class Controller(object):
	def __init__(self, model_name, search_space):
		self.model_name = model_name
		self.Search_space = search_space # HW search space
		self.Base_config = cp.ConfigParser()
		self.Base_config.read(os.path.join(home_path, 'debug_SimConfig.ini'), encoding='UTF-8')

		self.search_list = []
		self.area_list = []
		self.latency_list = []
		self.energy_list = []
		self.acc_list = []
	
	def cal_pni(self, pim_params):
		wl_weight = pim_params['wl_weight']
		adc_precision = pim_params['ADC_resolution']
		rram_vari = pim_params['RRAM_vari']
		subArray = pim_params['XBAR_size']
		cellbit = pim_params['RRAM_res']

		if 'rwkv' in self.model_name:
			model_name = 'rwkv-4-169m-pile'
		else:
			model_name = self.model_name
		
		sparsity = np.loadtxt('./prior_stat/{}/{}b_{}size_sparsity.txt'.format(model_name, wl_weight, subArray))
		act_sparsity = np.loadtxt('./prior_stat/{}/{}b_{}size_act_sparsity.txt'.format(model_name, wl_weight, subArray))
		var = np.loadtxt('./prior_stat/{}/{}b_{}size_var.txt'.format(model_name, wl_weight, subArray))
		base_var = np.loadtxt('./prior_stat/{}/base_var.txt'.format(model_name)).mean()

		total_qcn = 0
		ideal = 0
		weight_error = 0
		device_error = 0
		adc_error = 0

		for i in range(wl_weight-1):
			ideal += 2**(2*i)*var[i]
			adc_step_size = subArray*act_sparsity[i]/2**adc_precision
			device_error_distribution = norm(loc=0, scale=math.sqrt(subArray*var[i]*act_sparsity[i]*rram_vari**2))
			if rram_vari > 0:
				device_factor = scipy.integrate.quad(device_vari_function, -adc_step_size/2, adc_step_size/2, args=device_error_distribution)[0]
				device_factor /= (var[i]*act_sparsity[i]*rram_vari**2)*subArray
				device_factor = max(1-device_factor,0)
				device_error += 2**(2*i)*(var[i]*rram_vari**2)*device_factor
			adc_error += 2**(2*i)*((subArray*sparsity[i]**2/(2**(2*adc_precision)*12))*(1+rram_vari**2))

		weight_error = base_var*3/(2**(2*wl_weight))
		base_scale = 6*math.sqrt(base_var)/2**wl_weight
		weight_error = weight_error/(base_scale**2)
		# weight_error = ideal*3/(2**(2*wl_weight))

		error = weight_error + device_error + adc_error
		total_qcn = ideal/error

		return total_qcn

	
	def random_sample(self):
		pim_params = {}
		# Algorithm Properties
		pim_params['wl_weight'] = int(random.choice(self.Search_space['Algorithm_Part']['wl_weight'].split(',')))
		pim_params['wl_input'] = int(random.choice(self.Search_space['Algorithm_Part']['wl_input'].split(',')))
		# Hardware Properties
		pim_params['XBAR_size'] = int(random.choice(self.Search_space['XBAR_Part']['XBAR_size'].split(',')))
		pim_params['ADC_resolution'] = int(random.choice(self.Search_space['XBAR_Part']['ADC_resolution'].split(',')))
		pim_params['DAC_reuse'] = int(random.choice(self.Search_space['XBAR_Part']['DAC_reuse'].split(',')))
		pim_params['ADC_reuse'] = int(random.choice(self.Search_space['XBAR_Part']['ADC_reuse'].split(',')))
		# Device Properties
		pim_params['RRAM_res'] = int(random.choice(self.Search_space['Device_Part']['RRAM_res'].split(',')))
		pim_params['RRAM_vari'] = float(random.choice(self.Search_space['Device_Part']['RRAM_vari'].split(',')))
		# Other Properties
		pim_params['LUT_capacity'] = int(random.choice(self.Search_space['LUT_Part']['LUT_capacity'].split(',')))
		pim_params['LUT_num'] = int(random.choice(self.Search_space['LUT_Part']['LUT_num'].split(',')))
		pim_params['BUF_capacity'] = int(random.choice(self.Search_space['BUF_Part']['BUF_capacity'].split(',')))
		pim_params['BUF_num'] = int(random.choice(self.Search_space['BUF_Part']['BUF_num'].split(',')))
		pim_params['Digital_frequency'] = int(random.choice(self.Search_space['Other_Part']['Digital_frequency'].split(',')))

		return pim_params

	def search(self):
		for _ in range(10000): # iteration num bound is set to 10000
			params_sample = self.random_sample()
			if params_sample not in self.search_list:
				area, latency, energy, acc = self.evaluate(params_sample)
				self.search_list.append(params_sample)
				self.area_list.append(area)
				self.latency_list.append(latency)
				self.energy_list.append(energy)
				self.acc_list.append(acc)
		
	def evaluate(self, pim_params):
		wl_weight = pim_params['wl_weight']
		wl_input = pim_params['wl_input']
		# Hardware Properties
		XBAR_size = pim_params['XBAR_size']
		ADC_resolution = pim_params['ADC_resolution']
		DAC_reuse = pim_params['DAC_reuse']
		ADC_reuse = pim_params['ADC_reuse']
		# Device Properties
		RRAM_res = pim_params['RRAM_res']
		RRAM_vari = pim_params['RRAM_vari']
		# Other Properties
		LUT_capacity = pim_params['LUT_capacity']
		LUT_num = pim_params['LUT_num']
		BUF_capacity = pim_params['BUF_capacity']
		BUF_num = pim_params['BUF_num']
		Digital_frequency = pim_params['Digital_frequency']

		# model information [hidden_dimension, op_list, dimension_list, mode_list]
		D = model_to_dimension[self.model_name] if 'mamba' in self.model_name or 'rwkv' in self.model_name else None
		op_list, dimension_list, num_list, mode_list = get_ops_inf(params.model_name, D)

		# Simulation Initialization
		LUT = Look_Up_Table(LUT_capacity, LUT_num)
		BUF = Buffer(BUF_capacity)

		# SRAM predefined params
		SPE_latency = 0.09583
		SPE_energy = 24.04455e-4
		SPE_area = 78400
		SPE_capacity = 1

		# Area simulation part
		RRAM_area = 0
		RRAM_xbar_area = 0
		RRAM_dac_area = 0
		RRAM_adc_area = 0
		RRAM_digital_area = 0
		SRAM_area = 0
		LUT_area = LUT.calculate_area()
		BUF_area = BUF.calculate_area()

		# Latency simulation part
		RRAM_latency = 0
		RRAM_xbar_latency = 0
		RRAM_dac_latency = 0
		RRAM_adc_latency = 0
		RRAM_digital_latency = 0
		SRAM_latency = 0
		LUT_latency = 0
		BUF_latency = 0

		RRAM_energy = 0
		RRAM_xbar_energy = 0
		RRAM_dac_energy = 0
		RRAM_adc_energy = 0
		RRAM_digital_energy = 0
		SRAM_energy = 0
		LUT_energy = 0
		BUF_energy = 0

		MEM_access = 0

		# Analysis model operation iteratively
		for i in range(len(op_list)):
			op = op_list[i]
			dim = dimension_list[i]
			mode = mode_list[i]
			num = num_list[i]
			if isinstance(dim, float):
				dim = int(dim)
			elif isinstance(dim, int):
				pass
			else:
				dim = [int(dim[k]) for k in range(len(dim))]
			factor = 1
			if op in ['Linear', 'Conv1d', 'RoPE', 'Conv2d']:
				assert mode=='static', "Linear/Conv1d/Conv2d operation must be static mode, please check the config."
				if op == 'Conv2d':
					factor = int(dim[1])
					dim = (int(dim[0]),int(dim[-1]))*factor
				PE = ReRAM_ProcessElement(mode, pim_params, Digital_frequency, dim, 'RRAM')
				RRAM_area += num * PE.calculate_area()
				RRAM_xbar_area += num * PE.PE_xbar_area
				RRAM_dac_area += num * PE.PE_DAC_area
				RRAM_adc_area += num * PE.PE_ADC_area 
				RRAM_digital_area += num * PE.PE_digital_area
				RRAM_latency += factor*PE.calculate_read_latency()
				RRAM_xbar_latency += factor*PE.PE_xbar_read_latency
				RRAM_dac_latency += factor*PE.PE_DAC_read_latency
				RRAM_adc_latency += factor*PE.PE_ADC_read_latency
				RRAM_digital_latency += factor*PE.PE_digital_read_latency
				RRAM_energy += num * factor*PE.calculate_read_energy()
				RRAM_xbar_energy += num * factor*PE.PE_xbar_read_energy
				RRAM_dac_energy += num * factor*PE.PE_DAC_read_energy
				RRAM_adc_energy += num * factor*PE.PE_ADC_read_energy
				RRAM_digital_energy += num * factor*PE.PE_digital_read_energy
			elif op in ['EWM','RMSNorm','LayerNorm', 'SiLU', 'DWConv2d', 'BatchNorm']:
				if op == 'DWConv2d':
					next_dim = int(dimension_list[i+2][1])
					dim = next_dim**2*int(dim[0])*9 # kernel size is 3
				elif op == 'EWM':
					factor = 1
				else:
					factor = 2

				SRAM_energy += int(dim) / SPE_capacity * SPE_energy
				SRAM_latency += int(dim) / SPE_capacity * SPE_latency

				if op == 'SiLU':
					dim *= 3
					LUT_latency += LUT.calculate_latency(2*dim)
					LUT_energy += LUT.calculate_energy(2*dim)
			elif op in ['Sigmoid', 'SReLU','Softplus']:
				if op == 'Sigmoid':
					dim *= 3
				elif op == 'SReLU':
					dim *= 2
				elif op == 'Softplus':
					dim *= 2
				LUT_latency += LUT.calculate_latency(2*dim)
				LUT_energy += LUT.calculate_energy(2*dim)

			MEM_access += cal_mem_access(op, dim, num)
		
		# After iteratively analyzing
		# 1. simulate the buffer
		BUF_latency = BUF.calculate_read_latency(MEM_access/BUF_num)
		BUF_energy = 12*BUF.calculate_read_energy(MEM_access/BUF_num)

		# calculate total simulation results
		total_latency = (RRAM_latency + LUT_latency + BUF_latency)
		total_area = RRAM_area + LUT_area + BUF_area
		total_energy = (RRAM_energy + LUT_energy + BUF_energy)

		acc_metric = self.cal_pni(pim_params)

		return total_area, total_latency, total_energy, acc_metric

	

def main():
	device = torch.device("cuda")

	Search_space = cp.ConfigParser()
	Search_space.read(params.HWdes, encoding='UTF-8')
	Adam = Controller(params.model_name, Search_space)

	Adam.search()

	# filter
	index = []
	for i in range(len(Adam.area_list)):
		for j in range(len(Adam.area_list)):
			if i!=j and Adam.area_list[i]<Adam.area_list[j] and Adam.acc_list[i]>Adam.acc_list[j] and Adam.energy_list[i]*Adam.latency_list[i]<Adam.energy_list[j]*Adam.latency_list[j]:
				index.append(j)
	
	index = set(index)

	area_results = []
	latency_results = []
	energy_results = []
	pni_results = []
	for i in range(len(Adam.area_list)):
		if i not in index:
			area_results.append(Adam.area_list[i])
			latency_results.append(Adam.latency_list[i])
			energy_results.append(Adam.energy_list[i])
			pni_results.append(Adam.acc_list[i])


	area_results = np.array(area_results)
	latency_results = np.array(latency_results)
	energy_results = np.array(energy_results)
	pni_results = np.array(pni_results)
	edp_results = latency_results*energy_results

	############################################################
	# 1. 在不考虑accuracy的时候，找edp和AREA的帕罗托，在考虑accuracy的时候，需要加一条线画双y轴图 (保留全部参数变量)
	best_area = np.argsort(area_results).tolist()[:120]
	best_edp = np.argsort(edp_results).tolist()[:120]
	best_pni = np.argsort(pni_results).tolist()[:120]
	# best_area_edp_index = list(set(best_area) & set(best_edp))
	best_area_edp_index = list(set(best_area+best_edp))
	best_area_values = area_results[best_area_edp_index]
	best_edp_values = edp_results[best_area_edp_index]
	pni_values  = pni_results[best_area_edp_index]

	import ipdb; ipdb.set_trace()

	# save the results to txt for origin picture
	total_search_results = np.concatenate((best_edp_values[:,None], best_area_values[:,None], pni_values[:,None]),1)
	np.savetxt('./results/joint_search/{}_total_search_results.txt'.format(params.model_name), total_search_results)

	font=16
	size=(8, 8)
	fig = plt.figure(figsize=size)
	plt.scatter(best_edp_values, best_area_values, s=50, marker='.')
	plt.xticks(fontsize=font)
	plt.yticks(fontsize=font)
	plt.xlabel('$EDP$', fontsize=font)
	plt.ylabel('$Area$', fontsize=font)
	plt.grid(True)
	fig.tight_layout()
	fig.savefig('./results/joint_search/{}_edp_area.png'.format(params.model_name), dpi=600)

	import ipdb; ipdb.set_trace()


if __name__ == '__main__':
	main()