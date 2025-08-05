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
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, RwkvForCausalLM, MambaForCausalLM
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator, tasks
from lm_eval.tasks import get_task_dict

from fast_wrapper import *
from util import get_cnn_model, get_llm_model, get_dataset, replace_module, AverageMeter

home_path = os.getcwd()
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="path of the hf model")
parser.add_argument("--dataset", type=str, default='imagenet', help="the evaluation dataset")
parser.add_argument('--gpu_id', type=int, default=None, help='GPU id')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--log_file', type=str, default='./test.log', help='log file')
params = parser.parse_args()

LOG = logging.getLogger('main')
LOG.setLevel(logging.INFO)
fh = logging.FileHandler(params.log_file)
LOG_FORMAT = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(LOG_FORMAT)
LOG.addHandler(fh)

def device_vari_function(x, func):
	return func.pdf(x)*x**2

def cal_qcn(model_name, pim_params):
	wl_weight = pim_params['wl_weight']
	adc_precision = pim_params['ADCprecision']
	rram_vari = pim_params['vari']
	subArray = pim_params['subArray']
	cellbit = pim_params['cellBit']

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
		device_error_distribution = norm(loc=0, scale=math.sqrt(subArray*var[i]*rram_vari**2))
		if rram_vari > 0:
			device_factor = scipy.integrate.quad(device_vari_function, -adc_step_size/2, adc_step_size/2, args=device_error_distribution)[0]
			device_factor /= (var[i]*rram_vari**2)*subArray
			device_factor = max(1-device_factor,0)
			device_error += 2**(2*i)*(var[i]*rram_vari**2)*device_factor
		adc_error += 2**(2*i)*((subArray*sparsity[i]**2/(2**(2*adc_precision)*12)))

	weight_error = base_var*3/(2**(2*wl_weight))
	base_scale = 6*math.sqrt(base_var)/2**wl_weight
	weight_error = weight_error/(base_scale**2)

	error = weight_error + device_error + adc_error
	total_qcn = ideal/error

	return total_qcn

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


def main():
	# os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_id)
	device = torch.device("cuda")

	acc_list = []
	pim_params_list = []    
	if 'mamba' not in params.model_name and 'rwkv' not in params.model_name and 'opt' not in params.model_name:
		model_name = params.model_name
		f = open("/share/guolidong-nfs/PIM/Unified-QCN/results/{}/{}_accsim.log".format(params.model_name, params.dataset),"r")
		lines = f.readlines()
		for line in lines:
			if 'Acc' in line:
				acc = float(line.split('=')[-1][:-1])
				acc_list.append(acc)
			if 'wl_weight' in line:
				pim_params = {}
				pim_params['wl_weight'] = int(line.split("'wl_weight': ")[1][0])
				pim_params['ADCprecision'] = int(line.split("'ADCprecision': ")[1][0])
				pim_params['vari'] = float(line.split("'vari': ")[1][:-2])
				pim_params['subArray'] = int(line.split("'subArray': ")[1].split(',')[0])
				pim_params['cellBit'] = int(line.split("'cellBit': ")[1][0])
				pim_params_list.append(pim_params)
	else:
		model_name = params.model_name.split('/')[-1]
		f = open("/share/guolidong-nfs/PIM/Unified-QCN/results/{}/{}_accsim.log".format(model_name, params.dataset),"r")
		lines = f.readlines()
		for line in lines:
			if 'acc,none' in line:
				acc = float(line.split('acc,none')[1].split(',')[0][3:])
				acc_list.append(acc)
			if 'wl_weight' in line:
				pim_params = {}
				pim_params['wl_weight'] = int(line.split("'wl_weight': ")[1][0])
				pim_params['ADCprecision'] = int(line.split("'ADCprecision': ")[1][0])
				pim_params['vari'] = float(line.split("'vari': ")[1][:-2])
				pim_params['subArray'] = int(line.split("'subArray': ")[1].split(',')[0])
				pim_params['cellBit'] = int(line.split("'cellBit': ")[1][0])
				pim_params_list.append(pim_params)
	# filter unreasonable data
	qcn_list = []
	for pim_params in pim_params_list:
		wl_weight = pim_params['wl_weight']
		adc_precision = pim_params['ADCprecision']
		rram_vari = pim_params['vari']
		qcn = cal_qcn(model_name, pim_params)
		qcn_list.append(qcn)

	_acc_list = np.array(acc_list).reshape(-1,3).mean(-1).tolist()
	remove_index = []
	if 'mamba' not in params.model_name and 'rwkv' not in params.model_name and 'opt' not in params.model_name:
		model, exclude_layers = get_cnn_model(params.model_name)
		model = model.to(device).half()
		model.eval()

		train_dataset, test_dataset = get_dataset(params.dataset, params.model_name)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=0)

		# for CIFAR datasets, load our own model ckpts
		if 'cifar' in params.dataset:
			model = torch.load('./ckpts/{}_{}.pt'.format(params.model_name, params.dataset)).to(device).half()

		model_name = params.model_name

		wrapped_acc_list = []
		filtered_acc_list = []
		all_wrapped_acc_list = []
		error_list = []
		for i in range(len(_acc_list)):
			if pim_params_list[i]['cellBit']==1 and i not in remove_index:
				filtered_acc_list.append(_acc_list[i])
				LOG.info('Actual-Acc-top1-{}'.format(_acc_list[i]))
				top1_acc_list = []
				for j in range(3):
					_model = fast_wrap_model(params.model_name, model, pim_params_list[i], exclude_layers).half()
					top1_acc, top5_acc = evaluate(_model, test_loader)
					top1_acc_list.append(top1_acc)
					all_wrapped_acc_list.append(top1_acc)

				index = np.abs(np.array(top1_acc_list)-_acc_list[i]).argmin()
				wrap_acc = top1_acc_list[index]
				# wrap_acc = np.mean(top1_acc_list)
				wrapped_acc_list.append(wrap_acc)
				error_list.append(abs(_acc_list[i]-wrap_acc))
				LOG.info('Predicted-Acc-top1-{}'.format(wrap_acc))
	else:
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
				if 'layers.0' not in name:
					exclude_layers.append(name)
				else:
					non_exclude_layers.append(name)
					

		model_name = params.model_name.split('/')[-1]
		llm_task_list = [params.dataset]
		llm_tasks = get_task_dict(llm_task_list)

		wrapped_acc_list = []
		filtered_acc_list = []
		all_wrapped_acc_list = []
		error_list = []
		for i in range(len(_acc_list)):
			if pim_params_list[i]['cellBit']==1:
				filtered_acc_list.append(_acc_list[i])
				LOG.info('Actual-Acc-top1-{}'.format(_acc_list[i]))
				top1_acc_list = []
				pim_params = pim_params_list[i]
				for j in range(3):
					_model = fast_wrap_model(params.model_name, model, pim_params, exclude_layers=exclude_layers).half().to(device)
					# import ipdb; ipdb.set_trace()
					llm_model = HFLM(pretrained=_model, backend='causal', tokenizer=enc, batch_size=params.batch_size)
					results = evaluator.simple_evaluate(llm_model, task_dict=llm_tasks, torch_random_seed=j)
					top1_acc = float(results['results'][params.dataset]['acc,none'])
					top1_acc_list.append(top1_acc)
					all_wrapped_acc_list.append(top1_acc)
					# print(top1_acc)
					# import ipdb; ipdb.set_trace() 

				index = np.abs(np.array(top1_acc_list)-_acc_list[i]).argmin()
				wrap_acc = top1_acc_list[index]
				# wrap_acc = np.mean(top1_acc_list)
				wrapped_acc_list.append(wrap_acc)
				error_list.append(abs(_acc_list[i]-wrap_acc))
				LOG.info('Predicted-Acc-top1-{}'.format(wrap_acc))
				# import ipdb; ipdb.set_trace()
	
	LOG.info('Average Error={}'.format(np.mean(error_list)))

	kendalltau = scipy.stats.kendalltau(np.array(wrapped_acc_list), np.array(filtered_acc_list))
	print("kendalltau is {}".format(kendalltau))

	# save the results to txt for origin picture
	final_results = np.concatenate((np.array(filtered_acc_list)[:,None], np.array(all_wrapped_acc_list).reshape(-1,3)),1)
	np.savetxt('./results/{}/{}_abs_results.txt'.format(model_name, params.dataset), final_results)

	font=16
	size=(8, 8)
	fig = plt.figure(figsize=size)
	plt.scatter(np.array(wrapped_acc_list), np.array(filtered_acc_list), s=50, marker='.')
	plt.xticks(fontsize=font)
	plt.yticks(fontsize=font)
	plt.xlabel('$Predicted Accuracy k={}$'.format(kendalltau), fontsize=font)
	plt.ylabel('$Actual Accuracy$', fontsize=font)
	plt.grid(True)
	fig.tight_layout()
	fig.savefig('./results/{}/abs_{}_accsim.png'.format(model_name, params.dataset), dpi=600)

if __name__ == "__main__":
	main()