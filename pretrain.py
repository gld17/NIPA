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

from wrapper import *
from util import get_cnn_model, get_dataset, replace_module, AverageMeter

home_path = os.getcwd()
SimConfig_path = os.path.join(home_path, 'Search_space.ini')
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, help="path of the hf model")
parser.add_argument("--dataset", type=str, default='imagenet', help="the evaluation dataset")
parser.add_argument('--gpu_id', type=str, default=None, help='GPU id')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epochs', type=int, default=200, help='the number of finetune epochs')
parser.add_argument('--log_file', type=str, default='./test.log', help='log file')
params = parser.parse_args()

LOG = logging.getLogger('main')
LOG.setLevel(logging.INFO)
fh = logging.FileHandler(params.log_file)
sh = logging.StreamHandler()
LOG_FORMAT = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(LOG_FORMAT)
sh.setFormatter(LOG_FORMAT)
LOG.addHandler(sh)
LOG.addHandler(fh)

def train(args, model, train_loader, optim, epoch, criterion):
	model.train()
	losses = AverageMeter()
	total_images = 0
	correct_images = 0

	for i, data in enumerate(train_loader, 0):
		images, labels = data
		images = Variable(images.cuda())
		labels = Variable(labels.cuda())
		optim.zero_grad()
		outputs = model(images)
		_, predicts = torch.max(outputs.data, 1)
		loss = criterion(outputs, labels)
		losses.update(loss.item(), images.size(0))

		optim.zero_grad()
		loss.backward()
		optim.step()

		total_images += labels.size(0)
		correct_images += (predicts == labels).sum().item()

		if i % 2500 == 0:
			log_str = 'Epoch:{} Train:[{}/{}] Loss:{:.4f}({:.4f}) Acc: {:.5f} lr {:.6f}' \
					  .format(epoch, i, len(train_loader), losses.val, losses.avg, 100 * correct_images / total_images, optim.param_groups[0]['lr'])
			LOG.info(log_str)

def evaluate(model, dataloader):
	model.eval()
	total_images = 0
	correct_1 = 0
	correct_5 = 0
	with torch.no_grad():
		for i, data in enumerate(tqdm(dataloader, 0)):
			images, labels = data
			images = Variable(images.cuda())
			labels = Variable(labels.cuda())
			outputs = model(images)
			_, predicts_1 = torch.max(outputs.data, 1)
			_, predicts_5 = outputs.data.topk(5, 1, True, True)
			total_images += labels.size(0)
			correct_1 += (predicts_1 == labels).sum().item()
			correct_5 += (predicts_5 == labels.unsqueeze(1).expand_as(predicts_5)).sum().item()

	return 100 * correct_1 / total_images, 100*correct_5 / total_images

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = str(params.gpu_id)
	device = torch.device("cuda")
	
	model, exclude_layers = get_cnn_model(params.model_name, params.dataset)
	model = model.to(device)

	train_dataset, test_dataset = get_dataset(params.dataset, params.model_name)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=0)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=0)

	# for CIFAR datasets, load our own model ckpts
	if 'cifar' in params.dataset:
		model = replace_module(params.model_name, model, params.dataset)

	LOG.info('start training')
	criterion = nn.CrossEntropyLoss()
	ckpt_path = os.path.join('./ckpts', params.model_name+'_'+params.dataset+'.pt')
	cfg = dict(
		lr=1e-2,
		momentum=0.9,
		weight_decay=5e-4,
		gamma=0.2,
	)
	optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])   
	# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=280)
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=cfg['gamma'])

	if os.path.exists(ckpt_path):
		model = torch.load(ckpt_path)
	else:
		for epoch in range(params.epochs):
			train(params, model, train_loader, optimizer, epoch, criterion)
			lr_scheduler.step()
			if epoch%10==0:
				top1_acc, top5_acc = evaluate(model, test_loader)
				LOG.info('{}-th epochs: Acc-top1={} Acc-top5={}'.format(epoch, top1_acc, top5_acc))
		torch.save(model, ckpt_path)

	baseline_top1_acc, baseline_top5_acc = evaluate(model, test_loader)
	LOG.info('Baseline-acc-top1={} Baseline-acc-top5={}'.format(baseline_top1_acc, baseline_top5_acc))


	

if __name__ == "__main__":
	main()