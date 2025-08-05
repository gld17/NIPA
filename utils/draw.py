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
import numpy as np
import scipy


def main():
	qcn_list = []
	acc_list = []

	f = open("/share/guolidong-nfs/PIM/Unified-QCN/results/resnet18_cifar10_accsim.log","r")
	lines = f.readlines()
	for line in lines:
		if 'Acc' in line:
			acc = float(line.split('=')[-1][:-1])
			acc_list.append(acc)
		if 'QCN' in line:
			qcn = float(line.split(' ')[-1][:-1])
			qcn_list.append(qcn)
		# import ipdb; ipdb.set_trace()
	
	k = scipy.stats.kendalltau(np.array(qcn_list), np.array(acc_list))
	r = scipy.stats.pearsonr(np.array(qcn_list), np.array(acc_list))
	print("kendalltau is {}".format(k))
	print("pearsonr is {}".format(r))
		
	##### Draw ######
	# import ipdb; ipdb.set_trace()
	font=16
	size=(8, 7)
	fig = plt.figure(figsize=size)
	plt.scatter(np.log2(np.array(qcn_list)/np.min(qcn_list)), np.array(acc_list), s=50, marker='.')
	plt.xticks(fontsize=font)
	plt.yticks(fontsize=font)
	# plt.xlim(0,100000)
	plt.xlabel('$QCN-Metric$', fontsize=font)
	plt.ylabel('$Accuracy$', fontsize=font)
	plt.grid(True)
	fig.tight_layout()
	fig.savefig('./results/test.png', dpi=600)
	


	

if __name__ == "__main__":
	main()