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
import copy
from lm_eval.models.huggingface import HFLM
from lm_eval import evaluator, tasks
from lm_eval.tasks import get_task_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, RwkvForCausalLM, MambaForCausalLM

from fast_wrapper import *
from wrapper import *
from util import get_cnn_model, get_dataset, replace_module, AverageMeter

home_path = os.getcwd()
SimConfig_path = os.path.join(home_path, 'HW_config', 'Search_space.ini')
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

def add_error(model, vari):
	_model = copy.deepcopy(model)
	for name, module in _model.named_modules():
		if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
			weight = module.weight.detach()
			new_weight = weight + weight*torch.normal(0, math.sqrt(vari), weight.size()).to(weight.device)
			module.weight = nn.Parameter(new_weight.half())

	return _model

def evaluate(model, dataloader):
	total_images = 0
	correct_1 = 0
	correct_5 = 0
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
	device = torch.device("cuda")

	if 'mamba' in params.model_name or 'rwkv' in params.model_name:
		if 'mamba' in params.model_name:
			model = MambaForCausalLM.from_pretrained(params.model_name, device_map=device, torch_dtype=torch.float16)
			enc = AutoTokenizer.from_pretrained(params.model_name)
		elif 'rwkv' in params.model_name:
			model = RwkvForCausalLM.from_pretrained(params.model_name, device_map=device, torch_dtype=torch.float16)
			enc = AutoTokenizer.from_pretrained(params.model_name)

		llm_task_list = [params.dataset]
		llm_tasks = get_task_dict(llm_task_list)
		acc_list = []
		vari_list = []
		for vari in np.arange(0,3,0.05):
			vari_list.append(vari)
			average_acc_list = []
			for i in range(3):
				_model = add_error(model, vari)
				llm_model = HFLM(pretrained=_model, backend='causal', tokenizer=enc, batch_size=params.batch_size)
				results = evaluator.simple_evaluate(llm_model, task_dict=llm_tasks)
				acc = results['results'][params.dataset]['acc,none']
				average_acc_list.append(acc)
			acc_list.append(np.mean(average_acc_list))
			LOG.info('Accuracy of vari {} = {}'.format(vari, np.mean(average_acc_list)))
	else:
		model, exclude_layers = get_cnn_model(params.model_name)
		model = model.to(device).half()
		model.eval()

		train_dataset, test_dataset = get_dataset(params.dataset, params.model_name)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=0)

		# for CIFAR datasets, load our own model ckpts
		if 'cifar' in params.dataset:
			model = torch.load('./ckpts/{}_{}.pt'.format(params.model_name, params.dataset)).to(device).half()

		acc_list = []
		vari_list = []
		for vari in np.arange(0,3,0.05):
			vari_list.append(vari)
			average_acc_list = []
			for i in range(5):
				_model = add_error(model, vari)
				top1_acc, top5_acc = evaluate(_model, test_loader)
				average_acc_list.append(top1_acc)
			acc_list.append(np.mean(average_acc_list))
			LOG.info('Accuracy of vari {} = {}'.format(vari, np.mean(average_acc_list)))

	font=16
	size=(8, 8)
	fig = plt.figure(figsize=size)
	plt.scatter(np.array(vari_list), np.array(acc_list), s=50, marker='.')
	plt.xticks(fontsize=font)
	plt.yticks(fontsize=font)
	plt.xlabel('$Weight Variation$', fontsize=font)
	plt.ylabel('$Accuracy$', fontsize=font)
	plt.grid(True)
	fig.tight_layout()
	fig.savefig('./results/oracle/{}_{}.png'.format(params.model_name, params.dataset), dpi=600)
	# fig.savefig('./results/oracle/test.png', dpi=600)
	



	

if __name__ == "__main__":
	main()