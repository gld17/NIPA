import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import time
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

from wrapper import *
from util import get_cnn_model, get_dataset, replace_module, AverageMeter

home_path = os.getcwd()
SimConfig_path = os.path.join(home_path, 'HW_config', 'Search_space.ini')
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="path of the hf model")
parser.add_argument("--granularity", type=str, default='model')
parser.add_argument("--dataset", type=str, default='imagenet', help="the evaluation dataset")
parser.add_argument('--gpu_id', type=str, default=None, help='GPU id')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument("--HWdes", "--hardware_description", default=SimConfig_path)
params = parser.parse_args()

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
	os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_id)
	device = torch.device("cuda")

	if 'mamba' in params.model_name:
		from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
		from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, RwkvForCausalLM, MambaForCausalLM
		from lm_eval.models.huggingface import HFLM
		from lm_eval import evaluator, tasks
		from lm_eval.tasks import get_task_dict

		model_name = params.model_name.split('/')[-1]
		model = MambaForCausalLM.from_pretrained(params.model_name, device_map=device, torch_dtype=torch.float16)
		enc = AutoTokenizer.from_pretrained(params.model_name)

		exclude_layers = []
		# non_exclude_layers = []
		# for name, module in model.named_modules():
		# 	if 'layers.0' not in name and 'layers.23' not in name:
		# 		exclude_layers.append(name)
		# 	else:
		# 		non_exclude_layers.append(name)
		exclude_layers = ['lm_head']

		llm_task_list = [params.dataset]
		llm_tasks = get_task_dict(llm_task_list)

		Search_space = cp.ConfigParser()
		Search_space.read(params.HWdes, encoding='UTF-8')
		pim_params = random_sample(Search_space)

		_model = wrap_model(model, pim_params, exclude_layers=exclude_layers).to(device)
		llm_model = HFLM(pretrained=model, backend='causal', tokenizer=enc, batch_size=params.batch_size)
		_llm_model = HFLM(pretrained=_model, backend='causal', tokenizer=enc, batch_size=params.batch_size)

		def func():
			results = evaluator.simple_evaluate(llm_model, task_dict=llm_tasks)

		def wrap_func():
			results = evaluator.simple_evaluate(_llm_model, task_dict=llm_tasks)
	elif 'rwkv' in params.model_name:
		# from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
		from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, RwkvForCausalLM, MambaForCausalLM
		from lm_eval.models.huggingface import HFLM
		from lm_eval import evaluator, tasks
		from lm_eval.tasks import get_task_dict

		model_name = params.model_name.split('/')[-1]
		model = RwkvForCausalLM.from_pretrained(params.model_name, device_map=device, torch_dtype=torch.float16)
		enc = AutoTokenizer.from_pretrained(params.model_name)

		exclude_layers = []
		# non_exclude_layers = []
		# for name, module in model.named_modules():
		# 	if 'layers.0' not in name and 'layers.23' not in name:
		# 		exclude_layers.append(name)
		# 	else:
		# 		non_exclude_layers.append(name)
		exclude_layers = ['lm_head']

		llm_task_list = [params.dataset]
		llm_tasks = get_task_dict(llm_task_list)

		Search_space = cp.ConfigParser()
		Search_space.read(params.HWdes, encoding='UTF-8')
		pim_params = random_sample(Search_space)

		_model = wrap_model(model, pim_params, exclude_layers=exclude_layers).to(device)

		if params.granularity == 'block':
			block = model.rwkv.blocks[0]
			_block = _model.rwkv.blocks[0]
			hidden_states = torch.randn((256, 256, 768), dtype=torch.float16).to(device, non_blocking=True)
			def func():
				_ = block(hidden_states)

			def wrap_func():
				_ = _block(hidden_states)
		else:	
			llm_model = HFLM(pretrained=model, backend='causal', tokenizer=enc, batch_size=params.batch_size)
			_llm_model = HFLM(pretrained=_model, backend='causal', tokenizer=enc, batch_size=params.batch_size)

			def func():
				# results = evaluator.simple_evaluate(llm_model, task_dict=llm_tasks)
				results = evaluator.simple_evaluate(llm_model, tasks=llm_task_list)

			def wrap_func():
				# results = evaluator.simple_evaluate(_llm_model, task_dict=llm_tasks)
				results = evaluator.simple_evaluate(_llm_model, tasks=llm_task_list)


	else:
		model, exclude_layers = get_cnn_model(params.model_name)
		model = model.to(device).half()
		model.eval()

		train_dataset, test_dataset = get_dataset(params.dataset, params.model_name)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=0)

		# for CIFAR datasets, load our own model ckpts
		if 'cifar' in params.dataset:
			model = torch.load('./ckpts/{}_{}.pt'.format(params.model_name, params.dataset)).to(device).half()

		Search_space = cp.ConfigParser()
		Search_space.read(params.HWdes, encoding='UTF-8')
		pim_params = random_sample(Search_space)
		_model = wrap_model(model, pim_params, exclude_layers)

		def func():
			evaluate(model, test_loader)
		
		def wrap_func():
			wrap_func = evaluate(_model, test_loader)


	repetitions = 1
	pure_inf_time = 0
	torch.cuda.synchronize()

	# actual testing
	with torch.no_grad():
		for rep in range(repetitions):
			torch.cuda.synchronize()
			start_time = time.perf_counter()
			func()
			torch.cuda.synchronize()
			elapsed_time = time.perf_counter() - start_time
			pure_inf_time += elapsed_time

	mean_inf_time = pure_inf_time/repetitions
	print("Mean Inf Time of Actual Forward :{}ms".format(mean_inf_time*1e3))

	pure_inf_time = 0
	torch.cuda.synchronize()
	Search_space = cp.ConfigParser()
	Search_space.read(params.HWdes, encoding='UTF-8')
	pim_params = random_sample(Search_space)
	_model = wrap_model(model, pim_params, exclude_layers)

	# actual testing
	with torch.no_grad():
		torch.cuda.synchronize()
		start_time = time.perf_counter()
		wrap_func()
		torch.cuda.synchronize()
		elapsed_time = time.perf_counter() - start_time
		pure_inf_time += elapsed_time

	mean_inf_time = pure_inf_time
	print("Mean Inf Time of PIM Simulation Forward:{}ms".format(mean_inf_time*1e3))

if __name__ == "__main__":
	main()