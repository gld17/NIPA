import pickle
import copy
import torch
import torch.nn.functional as F
import math
from quant import *


def cal_sparsity_var_factor(weight, pim_params, index):
	_weight = copy.deepcopy(weight)
	subArray = pim_params['subArray']
	wl_weight = pim_params['wl_weight']
	sparsity = []
	act_sparsity = []
	var = []
	if index == wl_weight-1:
		return 0, 0
	
	factor = weight.shape[1]/subArray
	if _weight.shape[1] <= subArray:
		padding_num = subArray-_weight.shape[1]
		_weight = F.pad(_weight,(0,padding_num),'constant',0)
		_weight, w_scale = qw_tensor(_weight, wl_weight, subArray)
		X_decimal = _weight
		for k in range(wl_weight-1):
			remainder = torch.fmod(X_decimal, 2)
			X_decimal = torch.round((X_decimal-remainder)/2)
			ratio = ((remainder==1).sum(-1).max()+(remainder==-1).sum(-1).max())/(remainder>-2).sum(-1)[0]
			if k==index:
				var = torch.var(remainder,1).mean().item()
				sparsity = ratio.item()/math.sqrt(factor)
				# sparsity = ratio.item()
				act_sparsity = ratio.item()
				return sparsity, act_sparsity, var
	else:
		numSubArray = math.ceil(_weight.shape[1]/subArray)
		padding_num = numSubArray*subArray-_weight.shape[1]
		_weight = F.pad(_weight,(0,padding_num),'constant',0)

		_weight, w_scale = qw_tensor(_weight, wl_weight, subArray)
		_weight = _weight.reshape(-1, numSubArray, subArray)
		_sparsity = []
		_act_sparsity = []
		for xbar_index in range(numSubArray):
			X_decimal = _weight[:,xbar_index]
			for k in range(wl_weight-1):
				remainder = torch.fmod(X_decimal, 2)
				X_decimal = torch.round((X_decimal-remainder)/2)
				ratio = ((remainder==1).sum(-1).max()+(remainder==-1).sum(-1).max())/(remainder>-2).sum(-1)[0]
				if k==index:
					var.append(torch.var(remainder,1).mean().item())
					_sparsity.append(ratio.item()**2)
		
		# import ipdb; ipdb.set_trace()
		sparsity = math.sqrt(np.mean(_sparsity)*numSubArray**2/factor)
		act_sparsity = math.sqrt(np.mean(_sparsity))
		var = np.mean(var)

	return sparsity, act_sparsity, var

def cal_mem_access(op, dim, num):
	# calculate memory access including input and output
	if op in ['Linear', 'Conv1d', 'Conv2d']:
		mem_access = 1 * (dim[0] + dim[1]) # 1 means one-token decoding
	elif op in ['EWM', 'RMSNorm', 'LayerNorm', 'SiLU']:
		mem_access = 1 * (dim + dim + dim)
	elif op in ['Sigmoid', 'SReLU', 'Softplus']:
		mem_access = 1 * (dim + dim)
	else:
		mem_access = 0
	
	return num*mem_access

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

def get_ops_inf(model_name, D=None):
	if 'mamba' in model_name:
		assert D is not None
		op_list = ['RMSNorm', 'Linear', 'Conv1d', 'SiLU', 'Linear', 'Linear', 'Softplus', 'EWM', 'EWM', 'EWM', 'EWM', 'EWM','EWM','EWM','Linear']
		dimension_list = [(D),(D,D),(2*D,4),(2*D),(2*D,D/16+32),(D/16,2*D),(2*D),(32*D),(32*D),(32*D),(32*D),(32*D),(2*D),(2*D),(2*D,D)]
		num_list = [1,4,1,1,1,1,1,1,1,1,1,1,1,1,1]
		mode_list = ['act', 'static', 'static', 'act', 'static', 'static', 'None', 'dynamic', 'dynamic','dynamic','dynamic','dynamic','dynamic','dynamic', 'static']
	elif 'rwkv' in model_name:
		assert D is not None
		op_list = ['LayerNorm', 'EWM', 'Linear', 'Sigmoid', 'EWM', 'Linear', 'LayerNorm', 'EWM', 'Linear', 'SReLU', 'Linear', 'EWM']
		dimension_list = [(D),(6*D),(D,3*D),(D),(8*D),(D,D),(D),(4*D),(D,5*D),(D),(4*D,D),(D)]
		num_list = [1,1,1,1,1,1,1,1,1,1,1,1]
		mode_list = ['act', 'static', 'static', 'act', 'dynamic', 'static', 'act', 'static', 'static', 'act', 'static', 'dynamic']
	elif model_name == 'mobilenetv2':
		op_list = ['Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm','Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm',
				   'Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm','Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm',
				   'Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm','Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm',
				   'Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm','Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm',
				   'Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm','Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm',
				   'Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm','Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm',
				   'Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm','Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm',
				   'Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm','Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm',
				   'Conv2d','BatchNorm','DWConv2d','BatchNorm','Conv2d','BatchNorm','Conv2d','BatchNorm','Linear']
		dimension_list = [(27,224,32),32*(112**2),(32,112,32),32*(112**2),(32,112,16),16*(112**2),(16,112,96),96*(112**2),(96,112,96),96*(56**2),(96,56,24),24*(56**2),
						  (24,56,144),144*(56**2),(144,56,144),144*(56**2),(144,56,24),24*(56**2),(24,56,144),144*(56**2),(144,28,144),144*(28**2),(144,28,32),32*(28**2),
						  (32,28,192),192*(28**2),(192,28,192),192*(28**2),(192,28,32),32*(28**2),(32,28,192),192*(28**2),(192,28,192),192*(28**2),(192,28,32),32*(28**2),
						  (32,28,192),192*(28**2),(192,14,192),192*(14**2),(192,14,64),64*(14**2),(64,14,384),384*(14**2),(384,14,384),384*(14**2),(384,14,64),64*(14**2),
						  (64,14,384),384*(14**2),(384,14,384),384*(14**2),(384,14,64),64*(14**2),(64,14,384),384*(14**2),(384,14,384),384*(14**2),(384,14,64),64*(14**2),
						  (64,14,384),384*(14**2),(384,14,384),384*(14**2),(384,14,96),96*(14**2),(96,14,576),576*(14**2),(576,14,576),576*(14**2),(576,14,96),96*(14**2),
						  (96,14,576),576*(14**2),(576,14,576),576*(14**2),(576,14,96),96*(14**2),(96,14,576),576*(14**2),(576,14,576),576*(7**2),(576,7,160),160*(7**2),
						  (160,7,960),960*(7**2),(960,7,960),960*(7**2),(960,7,160),160*(7**2),(160,7,960),960*(7**2),(960,7,960),960*(7**2),(960,7,160),160*(7**2),
						  (160,7,960),960*(7**2),(960,7,960),960*(7**2),(960,7,320),320*(7**2),(320,7,1280),1280*(7**2),(1280,1000)]
		num_list = [1 for i in range(len(op_list))]
		mode_list = ['static','act','static','act','static','act','static','act','static','act','static','act',
					 'static','act','static','act','static','act','static','act','static','act','static','act',
					 'static','act','static','act','static','act','static','act','static','act','static','act',
					 'static','act','static','act','static','act','static','act','static','act','static','act',
					 'static','act','static','act','static','act','static','act','static','act','static','act',
					 'static','act','static','act','static','act','static','act','static','act','static','act',
					 'static','act','static','act','static','act','static','act','static','act','static','act',
					 'static','act','static','act','static','act','static','act','static','act','static','act',
					 'static','act','static','act','static','act','static','act','static']
	elif model_name == 'efficientnetv2m':
		op_list = []
		dimension_list = []
		mode_list = []

		with open('/share/guolidong-nfs/PIM/PN[TCAD]/op_list.pkl', 'rb') as f:
			op_list = pickle.load(f)
		with open('/share/guolidong-nfs/PIM/PN[TCAD]/mode_list.pkl', 'rb') as f:
			mode_list = pickle.load(f)
		with open('/share/guolidong-nfs/PIM/PN[TCAD]/dimension_list.pkl', 'rb') as f:
			dimension_list = pickle.load(f)

		num_list = [1 for i in range(len(op_list))]

	return op_list, dimension_list, num_list, mode_list