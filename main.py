# Need to solve proxy connection problem
# export HTTP_PROXY=http://127.0.0.1:7890
# export HTTPS_PROXY=http://127.0.0.1:7890

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
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator, tasks
from lm_eval.tasks import get_task_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, RwkvForCausalLM, MambaForCausalLM

from wrapper import *
from util import get_cnn_model, get_llm_model, get_dataset, replace_module, seed_everything

home_path = os.getcwd()
SimConfig_path = os.path.join(home_path, 'HW_config', 'Search_space.ini')
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="path of the hf model")
parser.add_argument("--model", type=str, help="Name of model e.g. `hf`, only for LLM")
parser.add_argument("--dataset", type=str, default='imagenet', help="the evaluation dataset")
parser.add_argument("--tasks", default=None, help="Available Tasks for LLM")
parser.add_argument('--gpu_id', type=int, default=None, help='GPU id')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--sample_num', type=int, default=200, help='the number of samples')
parser.add_argument('--cycle', type=int, default=3, help='the cycle number of one sample')
parser.add_argument("--HWdes", "--hardware_description", default=SimConfig_path)
parser.add_argument('--log_file', type=str, default='./test.log', help='log file')
parser.add_argument('--exclude_layer', type=str, help='exclude layers')
params = parser.parse_args()

LOG = logging.getLogger('main')
LOG.setLevel(logging.INFO)
fh = logging.FileHandler(params.log_file)
LOG_FORMAT = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(LOG_FORMAT)
LOG.addHandler(fh)

def evaluate(model, dataloader):
	total_images = 0
	correct_1 = 0
	correct_5 = 0
	iter = 0
	with torch.no_grad():
		for i, data in enumerate(tqdm(dataloader, 0)):
			images, labels = data
			images = Variable(images.cuda()).half()
			labels = Variable(labels.cuda()).half()
			outputs = model(images)
			_, predicts_1 = torch.max(outputs.data, 1)
			_, predicts_5 = outputs.data.topk(5, 1, True, True)
			total_images += labels.size(0)
			correct_1 += (predicts_1 == labels).sum().item()
			correct_5 += (predicts_5 == labels.unsqueeze(1).expand_as(predicts_5)).sum().item()

	return 100 * correct_1 / total_images, 100*correct_5 / total_images

def random_sample(Search_space):
	pim_params = {}
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
	pim_params['laplace_b'] = float(random.choice(Search_space['Device_Level']['laplace_b'].split(',')))
	pim_params['ind_vari'] = float(random.choice(Search_space['Device_Level']['ind_vari'].split(',')))
	pim_params['SAF_prop'] = float(random.choice(Search_space['Device_Level']['SAF_prop'].split(',')))

	return pim_params

def main():
	# os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu_id
	device = torch.device("cuda")

	Search_space = cp.ConfigParser()
	Search_space.read(params.HWdes, encoding='UTF-8')

	acc_list = []
	qcn_list = []
	_params_list = []
	
	for i in range(params.sample_num):
		pim_params = random_sample(Search_space)
		# TODO: add non-gaussian exps
		if pim_params not in _params_list:
			_params_list.append(pim_params)
		else:
			continue

	# import ipdb; ipdb.set_trace()
	
	if 'mamba' in params.model_name or 'rwkv' in params.model_name or 'opt' in params.model_name or 'llama' in params.model_name:
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
		elif 'Llama' in params.model_name:
            enc = AutoTokenizer.from_pretrained(params.model_name)
			model = AutoModelForCausalLM.from_pretrained(params.model_name, device_map='balanced').half()
			exclude_layers = []
			non_exclude_layers = []
			for name, module in model.named_modules():
				if 'layers.0' not in name:
					exclude_layers.append(name)
				else:
					non_exclude_layers.append(name)
		elif 'opt' in params.model_name:
			model = AutoModelForCausalLM.from_pretrained(params.model_name, device_map='auto').half()
			enc = AutoTokenizer.from_pretrained(params.model_name)
			exclude_layers = []
			non_exclude_layers = []
			for name, module in model.named_modules():
				if params.exclude_layer not in name:
					exclude_layers.append(name)
				else:
					non_exclude_layers.append(name)

		llm_task_list = [params.dataset]
		# llm_tasks = get_task_dict(llm_task_list)
		for i in range(len(_params_list)):
			pim_params = _params_list[i]
			LOG.info("The {}-th PIM parameters".format(i))
			LOG.info(pim_params)
			# wrap the model with considering the weight precision / ADC precision / device variations / XBAR size / Cell variation
			for j in range(params.cycle):
				_model = wrap_model(model, pim_params, exclude_layers=exclude_layers)
				llm_model = HFLM(pretrained=_model, backend='causal', tokenizer=enc, batch_size=params.batch_size)
				results = evaluator.simple_evaluate(llm_model, tasks=llm_task_list, torch_random_seed=j)
				LOG.info(results['results'])
	else:
		model, exclude_layers = get_cnn_model(params.model_name)
		model = model.to(device).half()
		model.eval()

		train_dataset, test_dataset = get_dataset(params.dataset, params.model_name)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=4)

		# for CIFAR datasets, load our own model ckpts
		if 'cifar' in params.dataset:
			model = torch.load('./ckpts/{}_{}.pt'.format(params.model_name, params.dataset)).to(device).half()

		for i in range(len(_params_list)):
			pim_params = _params_list[i]
			LOG.info("The {}-th PIM parameters".format(i))
			LOG.info(pim_params)
			# wrap the model with considering the weight precision / ADC precision / device variations / XBAR size / Cell variation
			for j in range(params.cycle):
				_model = wrap_model(model, pim_params, [])
				top1_acc, top5_acc = evaluate(_model, test_loader)
				LOG.info('Acc-top1={}'.format(top1_acc))
	

if __name__ == "__main__":
	main()