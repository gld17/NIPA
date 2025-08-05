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
SimConfig_path = os.path.join(home_path, 'HW_config', 'Search_space_hwinsight.ini')
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="path of the hf model")
parser.add_argument("--dataset", type=str, default='imagenet', help="the evaluation dataset")
parser.add_argument("--log_file", type=str, default='./results/insight/hw', help="the evaluation dataset")
parser.add_argument('--gpu_id', type=int, default=None, help='GPU id')
parser.add_argument('--cycle', type=int, default=3, help='the cycle number of one sample')
parser.add_argument("--HWdes", "--hardware_description", default=SimConfig_path)
parser.add_argument('--sample_num', type=int, default=200, help='the number of samples')
parser.add_argument('--device_only', action='store_true', help='whether to model device error independently')
params = parser.parse_args()

LOG = logging.getLogger('main')
LOG.setLevel(logging.INFO)
fh = logging.FileHandler(params.log_file)
LOG_FORMAT = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(LOG_FORMAT)
LOG.addHandler(fh)

def device_vari_function(x, func):
	return func.pdf(x)*x**2

def calculate_variance(prop, var_w, abs_mean_w, var_d):
	total_variance = prop * var_w + prop * var_w + (1-2*prop)*var_d
	
	return total_variance

def cal_qcn(model_name, pim_params, device_only=False):
	wl_weight = pim_params['wl_weight']
	adc_precision = pim_params['ADCprecision']
	vari_type = pim_params['vari_type']
	rram_vari = pim_params['vari']
	rram_ind_vari = pim_params['ind_vari']
	SAF_prop = pim_params['SAF_prop']
	subArray = pim_params['subArray']
	cellbit = pim_params['cellBit']

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
		device_error_distribution = norm(loc=0, scale=math.sqrt(subArray*var[i]*rram_vari**2))
		if rram_vari > 0:
			device_factor = scipy.integrate.quad(device_vari_function,0, adc_step_size, args=device_error_distribution)[0]
			device_factor /= (var[i]*rram_vari**2)*subArray
			device_factor = max(1-device_factor,0)
			# device_error += 2**(2*i)*(var[i]*rram_vari**2)*device_factor
			if vari_type == 'state_dependent':
				vari = calculate_variance(SAF_prop,var[i],abs_mean[i],var[i]*rram_vari**2)
				device_error += 2**(2*i)*vari*device_factor
			else:
				vari = calculate_variance(SAF_prop,var[i],abs_mean[i],rram_ind_vari**2)
				device_error += 2**(2*i)*vari*device_factor
		adc_error += 2**(2*i)*((subArray*sparsity[i]**2/(2**(2*adc_precision)*12)))/math.sqrt(act_sparsity[i])

	weight_error = base_var*3/(2**(2*wl_weight))
	base_scale = 6*math.sqrt(base_var)/2**wl_weight
	weight_error = weight_error/(base_scale**2)
	weight_error = 3*ideal/(2**(2*wl_weight))

	if device_only:
		error = device_error+1e-2
	else:
		error = weight_error + device_error + adc_error
	total_qcn = ideal/error
	return math.sqrt(total_qcn)


def random_sample(Search_space):
	pim_params = {}
	# import ipdb; ipdb.set_trace()
	# Algorithm Properties
	pim_params['wl_weight'] = int(random.choice(Search_space['Algorithm_Level']['wl_weight'].split(',')))
	pim_params['wl_input'] = int(random.choice(Search_space['Algorithm_Level']['wl_input'].split(',')))
	# Hardware Properties
	pim_params['subArray'] = int(random.choice(Search_space['Hardware_Level']['subArray'].split(',')))
	pim_params['ADCprecision'] = int(random.choice(Search_space['Hardware_Level']['ADCprecision'].split(',')))
	# Device Properties
	pim_params['cellBit'] = int(random.choice(Search_space['Device_Level']['cellBit'].split(',')))
	pim_params['onoffratio'] = int(random.choice(Search_space['Device_Level']['onoffratio'].split(',')))
	# if do not run the device retention / conductance variation effects, set args.vari=0, args.v=0
	pim_params['vari_type'] = str(random.choice(Search_space['Device_Level']['vari_type'].split(',')))
	pim_params['vari'] = float(random.choice(Search_space['Device_Level']['vari'].split(',')))
	pim_params['ind_vari'] = float(random.choice(Search_space['Device_Level']['ind_vari'].split(',')))
	pim_params['SAF_prop'] = float(random.choice(Search_space['Device_Level']['SAF_prop'].split(',')))

	return pim_params


def main():
	device = torch.device("cuda")

	Search_space = cp.ConfigParser()
	Search_space.read(params.HWdes, encoding='UTF-8')

	acc_list = []
	qcn_list = []
	_params_list = []
	
	for i in range(params.sample_num):
		pim_params = random_sample(Search_space)
		if pim_params not in _params_list:
			_params_list.append(pim_params)
		else:
			continue

	qcn_list = []
	for pim_params in _params_list:
		qcn = cal_qcn(params.model_name, pim_params, device_only=params.device_only)
		LOG.info(pim_params)
		LOG.info(qcn)

if __name__ == "__main__":
	main()