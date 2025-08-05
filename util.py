import os
import torch
import random
import numpy as np
from torchvision import transforms
import torchvision
import torchvision.models as models
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, RwkvForCausalLM, MambaForCausalLM

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
		
def get_module_by_name_suffix(model, module_name: str):
	for name, module in model.named_modules():
		if name.endswith(module_name):
			return module

def get_module_by_name(model, module_name):
	"""
	Get a module specified by its module name

	Parameters
	----------
	model : pytorch model
		the pytorch model from which to get its module
	module_name : str
		the name of the required module

	Returns
	-------
	module, module
		the parent module of the required module, the required module
	"""
	name_list = module_name.split(".")
	for name in name_list[:-1]:
		model = getattr(model, name)
	leaf_module = getattr(model, name_list[-1])
	return model, leaf_module

def get_cnn_model(model_name, dataset='imagenet'):
	if dataset == 'imagenet':
		if model_name == 'resnet18':
			model = models.resnet18(pretrained=True)
			exclude_layers = ['fc']
		elif model_name == 'resnet34':
			model = models.resnet34(pretrained=True)
			exclude_layers = ['fc']
		elif model_name == 'resnet50':
			model = models.resnet50(pretrained=True)
			exclude_layers = ['fc']
		elif model_name == 'vgg16':
			model = models.vgg16(pretrained=True)
			exclude_layers = ['classifier.6']
		elif model_name == 'mobilenetv2':
			model = models.mobilenet_v2(pretrained=True)
			exclude_layers = ['classifier.1']
		elif model_name == 'efficientnetv2m':
			model = models.efficientnet_v2_m(pretrained=True)
			exclude_layers = ['classifier.1']
	else:
		if model_name == 'resnet18':
			model = models.resnet18()
			exclude_layers = ['fc']
		elif model_name == 'resnet34':
			model = models.resnet34()
			exclude_layers = ['fc']
		elif model_name == 'vgg16':
			model = models.vgg16(pretrained=True)
			exclude_layers = ['classifier.6']
		elif model_name == 'resnet50':
			model = models.resnet50()
			exclude_layers = ['fc']
		elif model_name == 'mobilenetv2':
			model = models.mobilenet_v2()
			exclude_layers = ['classifier.1']
		elif model_name == 'efficientnetv2m':
			model = models.efficientnet_v2_m()
			exclude_layers = ['classifier.1']
	return model, exclude_layers

def get_llm_model(model_name):
	if 'mamba' in model_name:
		model = MambaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
		enc = AutoTokenizer.from_pretrained(model_name)
		
		exclude_layers = []
		non_exclude_layers = []
		for name, module in model.named_modules():
			if 'layers.23' not in name and 'x_proj' not in name and 'out_proj' not in name:
				exclude_layers.append(name)
			else:
				non_exclude_layers.append(name)
	elif 'rwkv' in model_name:
		model = RwkvForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
		enc = AutoTokenizer.from_pretrained(model_name)

		exclude_layers = []
		non_exclude_layers = []
		for name, module in model.named_modules():
			if 'blocks.11' not in name:
				exclude_layers.append(name)
			else:
				non_exclude_layers.append(name)
	
	return model, enc, exclude_layers
	

def replace_module(model_name, model, dataset):
	if dataset == 'cifar10':
		if 'resnet' in model_name:
			super_module, leaf_module = get_module_by_name(model, 'fc')    
			classifier = torch.nn.Linear(in_features=leaf_module.in_features, out_features=10, bias=leaf_module.bias is not None).cuda()
			setattr(super_module, 'fc', classifier)
		elif 'vgg' in model_name:
			super_module, leaf_module = get_module_by_name(model, 'classifier.6')    
			classifier = torch.nn.Linear(in_features=leaf_module.in_features, out_features=10, bias=leaf_module.bias is not None).cuda()
			setattr(super_module, '6', classifier)
		elif 'mobilenet' in model_name or 'efficientnet' in model_name:
			super_module, leaf_module = get_module_by_name(model, 'classifier.1')    
			classifier = torch.nn.Linear(in_features=leaf_module.in_features, out_features=10, bias=leaf_module.bias is not None).cuda()
			setattr(super_module, '1', classifier)
	elif dataset == 'cifar100':
		if 'resnet' in model_name:
			super_module, leaf_module = get_module_by_name(model, 'fc')    
			classifier = torch.nn.Linear(in_features=leaf_module.in_features, out_features=100, bias=leaf_module.bias is not None).cuda()
			setattr(super_module, 'fc', classifier)
		elif 'vgg' in model_name:
			super_module, leaf_module = get_module_by_name(model, 'classifier.6')    
			classifier = torch.nn.Linear(in_features=leaf_module.in_features, out_features=100, bias=leaf_module.bias is not None).cuda()
			setattr(super_module, '6', classifier)
		elif 'mobilenet' in model_name or 'efficientnet' in model_name:
			super_module, leaf_module = get_module_by_name(model, 'classifier.1')    
			classifier = torch.nn.Linear(in_features=leaf_module.in_features, out_features=100, bias=leaf_module.bias is not None).cuda()
			setattr(super_module, '1', classifier)
	
	return model


def get_dataset(dataset_name, model_name):
	if dataset_name == 'cifar100':
		# Load test dataset
		transform_test = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
		transform_train = transforms.Compose([
			#transforms.ToPILImage(),
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(15),
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
		test_dataset = torchvision.datasets.CIFAR100(root='/share/guolidong-local/CIFAR100', train=False, download=True, transform=transform_test)
		train_dataset = torchvision.datasets.CIFAR100(root='/share/guolidong-local/CIFAR100', train=True, download=True, transform=transform_train)
	elif dataset_name == 'cifar10':
		# Load test dataset
		transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		])
		test_dataset = torchvision.datasets.CIFAR10(root='/share/guolidong-local/CIFAR10', train=False, download=True, transform=transform)
		train_dataset = torchvision.datasets.CIFAR10(root='/share/guolidong-local/CIFAR10', train=True, download=True, transform=transform)
	elif dataset_name == 'imagenet':
		train_dataset = None # no need to train on imagenet
		testdir = os.path.join('/share/public-local/datasets/ILSVRC2012/', 'val')
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										std=[0.229, 0.224, 0.225])
		if model_name == 'efficientnetv2s':
			test_dataset = torchvision.datasets.ImageFolder(
					testdir,
					transforms.Compose([
						transforms.Resize(384),
						transforms.CenterCrop(384),
						transforms.ToTensor(),
						normalize,
					]))
		elif model_name == 'efficientnetv2m':
			test_dataset = torchvision.datasets.ImageFolder(
					testdir,
					transforms.Compose([
						transforms.Resize(480),
						transforms.CenterCrop(480),
						transforms.ToTensor(),
						normalize,
					]))
		elif model_name == 'efficientnetv2l':
			normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
											 std=[0.5, 0.5, 0.5])
			test_dataset = torchvision.datasets.ImageFolder(
					testdir,
					transforms.Compose([
						transforms.Resize(480),
						transforms.CenterCrop(480),
						transforms.ToTensor(),
						normalize,
					]))
		elif model_name in ['mobilenetv2', 'mobilenetv3l']:
			test_dataset = torchvision.datasets.ImageFolder(
					testdir,
					transforms.Compose([
						transforms.Resize(232),
						transforms.CenterCrop(224),
						transforms.ToTensor(),
						normalize,
					]))
		else:
			test_dataset = torchvision.datasets.ImageFolder(
					testdir,
					transforms.Compose([
						transforms.Resize(256),
						transforms.CenterCrop(224),
						transforms.ToTensor(),
						normalize,
					]))
	
	return train_dataset, test_dataset