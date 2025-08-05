import math
import copy
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from quant import *

def _setattr(model, name, module):
	name_list = name.split(".")
	for name in name_list[:-1]:
		model = getattr(model, name)
	setattr(model, name_list[-1], module)

def apply_SAF(tensor, prop=0.1):
	proba = torch.rand_like(tensor).to(tensor.device)  # 生成 [0,1] 区间的随机数
	tensor = torch.where(prob < p_zero, torch.tensor(0.0, device=tensor.device), tensor)
	tensor = torch.where(prob > 1 - p_one, torch.tensor(1.0, device=tensor.device), tensor)
	return tensor

class QConv2d(nn.Module):
	def __init__(self, name, module, pim_params):
		super().__init__()
		self.name = name
		self.wl_weight = pim_params['wl_weight']
		self.wl_input = pim_params['wl_input']
		self.subArray = pim_params['subArray']
		self.ADCprecision = pim_params['ADCprecision']
		self.cellBit = pim_params['cellBit']
		self.onoffratio = pim_params['onoffratio']
		self.vari_type = pim_params['vari_type']
		self.vari = pim_params['vari']
		self.laplace_b = pim_params['laplace_b']
		self.ind_vari = pim_params['ind_vari']
		self.SAF_prop = pim_params['SAF_prop']

		self.weight = module.weight
		self.kernel_size = module.kernel_size
		self.bias = module.bias
		self.stride = module.stride
		self.padding = module.padding
		self.dilation = module.dilation
		self.groups = module.groups

		self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation)
		self.apply_SAF()
	
	@torch.no_grad()
	def apply_SAF(self):
		weight = self.weight.reshape(self.weight.shape[0],-1)
		self.SA0_mask = []
		self.SA1_mask = []
		for k in range (int(self.wl_weight/self.cellBit)-1):
			if weight.shape[1] <= self.subArray:
				prob = torch.rand_like(weight)  # 生成 [0,1] 区间的随机数
				self.SA0_mask.append(torch.where(prob < self.SAF_prop))
				self.SA1_mask.append(torch.where(prob > 1 - self.SAF_prop))
			else:
				# need to divide to different subArray
				numSubArray = math.ceil(weight.shape[1]/self.subArray)
				padding_flag = numSubArray*self.subArray > weight.shape[1]
				if padding_flag:
					padding_num = numSubArray*self.subArray-weight.shape[1]
					weight = F.pad(weight,(0,padding_num),'constant',0)
					weight = weight.reshape(-1,numSubArray,self.subArray)
				else:
					weight = weight.reshape(-1,numSubArray,self.subArray)
				prob = torch.rand_like(weight)  # 生成 [0,1] 区间的随机数
				self.SA0_mask.append(torch.where(prob < self.SAF_prop))
				self.SA1_mask.append(torch.where(prob > 1 - self.SAF_prop))

	@torch.no_grad()
	def forward(self, input):
		outputOrignal_shape = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups).shape
		output = torch.zeros(outputOrignal_shape).cuda().half()

		bitWeight = int(self.wl_weight)
		bitActivation = int(self.wl_input)

		# set parameters for Hardware Inference
		onoffratio = self.onoffratio
		upper = 1
		lower = 1/onoffratio
		cellRange = 2**self.cellBit

		# first convert conv operation 
		input = self.unfold(input).permute(0,2,1).reshape(-1,self.unfold(input).shape[1])
		weight = self.weight.reshape(self.weight.shape[0],-1)

		if weight.shape[1] <= self.subArray:
			weight, w_scale = qw_tensor(weight, self.wl_weight, weight.shape[1])
			# quantize input into binary sequence
			inputQ, a_scale = qa_tensor(input, self.wl_input, per_tensor=True)
			for z in range(bitActivation-1):
				inputB = torch.fmod(inputQ, 2)
				inputQ = torch.round((inputQ-inputB)/2)
				# after get the spacial kernel, need to transfer floating weight [-1, 1] to binarized ones
				X_decimal = weight
				outputD = torch.zeros(outputOrignal_shape).cuda().half()
				for k in range (int(bitWeight/self.cellBit)-1):
					remainder = torch.fmod(X_decimal, cellRange)
					X_decimal = torch.round((X_decimal-remainder)/cellRange)
					if self.vari_type == 'state_dependent':
						variation = torch.normal(0, self.vari, remainder.size()).cuda().half()
						remainderQ = remainder + remainder*variation 
					elif self.vari_type == 'laplace':
						laplace_dist = torch.distributions.Laplace(0, self.laplace_b)
						variation = laplace_dist.sample(remainder.size()).cuda().half()
						remainderQ = remainder + remainder*variation
					elif self.vari_type == 'state_independent':
						ind_variation = torch.normal(0, self.ind_vari, remainder.size()).cuda().half()
						remainderQ = remainder + ind_variation
					else:
						import ipdb; ipdb.set_trace()
					remainderQ[self.SA0_mask[k]] = lower
					remainderQ[self.SA1_mask[k]] = torch.sign(remainderQ[self.SA1_mask[k]])
					outputPartial= F.linear(inputB, remainderQ)
					# Add ADC quanization effects here !!!
					outputPartialQ = LinearQuantizeOut(outputPartial, self.ADCprecision)*w_scale.permute(1,0)
					scaler = cellRange**k
					outputD = outputD + (outputPartialQ*scaler).reshape(outputOrignal_shape[0],outputOrignal_shape[2],outputOrignal_shape[3],outputOrignal_shape[1]).permute(0,3,1,2)
				scalerIN = 2**z
				output += outputD * scalerIN * a_scale
		else:
			inputQ, a_scale = qa_tensor(input, self.wl_input, per_tensor=True)
			# need to divide to different subArray
			numSubArray = math.ceil(weight.shape[1]/self.subArray)
			padding_flag = numSubArray*self.subArray > weight.shape[1]
			if padding_flag:
				padding_num = numSubArray*self.subArray-weight.shape[1]
				inputQ = F.pad(inputQ,(0,padding_num),'constant',0).reshape(-1,numSubArray,self.subArray)
				weight = F.pad(weight,(0,padding_num),'constant',0)
				weight, w_scale = qw_tensor(weight, self.wl_weight, self.subArray)
				weight = weight.reshape(-1,numSubArray,self.subArray)
			else:
				weight, w_scale = qw_tensor(weight, self.wl_weight, self.subArray)
				weight = weight.reshape(-1,numSubArray,self.subArray)
				inputQ = inputQ.reshape(-1,numSubArray,self.subArray)
			for z in range(bitActivation-1):
				outputD = torch.zeros(outputOrignal_shape).cuda().half()

				inputB = torch.fmod(inputQ, 2)
				inputQ = torch.round((inputQ-inputB)/2)

				X_decimal = weight
				for k in range(int(bitWeight/self.cellBit)-1):
					remainder = torch.fmod(X_decimal, cellRange)
					X_decimal = torch.round((X_decimal-remainder)/cellRange)
					if self.vari_type == 'state_dependent':
						variation = torch.normal(0, self.vari, remainder.size()).cuda().half()
						remainderQ = remainder + remainder*variation 
					elif self.vari_type == 'laplace':
						laplace_dist = torch.distributions.Laplace(0, self.laplace_b)
						variation = laplace_dist.sample(remainder.size()).cuda().half()
						remainderQ = remainder + remainder*variation
					elif self.vari_type == 'state_independent':
						ind_variation = torch.normal(0, self.ind_vari, remainder.size()).cuda().half()
						remainderQ = remainder + ind_variation
					else:
						import ipdb; ipdb.set_trace()
					remainderQ[self.SA0_mask[k]] = lower
					remainderQ[self.SA1_mask[k]] = torch.sign(remainderQ[self.SA1_mask[k]])
					outputPartial = torch.matmul(inputB.permute(1,0,2),remainderQ.permute(1,2,0))
					# Add ADC quanization effects here !!!
					outputPartialQ = (LinearQuantizeOut(outputPartial, self.ADCprecision)*w_scale.reshape(-1,numSubArray).permute(1,0)[:,None,:]).sum(0)
					scaler = cellRange**k
					outputD = outputD + (outputPartialQ*scaler).reshape(outputOrignal_shape[0],outputOrignal_shape[2],outputOrignal_shape[3],outputOrignal_shape[1]).permute(0,3,1,2)
				scalerIN = 2**z
				output += outputD * scalerIN * a_scale

		if self.bias is not None:
			output += self.bias[None,:,None,None]

		return output

class QLinear(nn.Module):
	def __init__(self, name, module, pim_params):
		super().__init__()
		self.name = name
		self.wl_weight = pim_params['wl_weight']
		self.wl_input = pim_params['wl_input']
		self.subArray = pim_params['subArray']
		self.ADCprecision = pim_params['ADCprecision']
		self.cellBit = pim_params['cellBit']
		self.onoffratio = pim_params['onoffratio']
		self.vari_type = pim_params['vari_type']
		self.vari = pim_params['vari']
		self.laplace_b = pim_params['laplace_b']
		self.ind_vari = pim_params['ind_vari']
		self.SAF_prop = pim_params['SAF_prop']

		self.weight = module.weight
		self.bias = module.bias

		self.apply_SAF()

	@torch.no_grad()
	def apply_SAF(self):
		weight = self.weight.reshape(self.weight.shape[0],-1)
		self.SA0_mask = []
		self.SA1_mask = []
		for k in range (int(self.wl_weight/self.cellBit)-1):
			if weight.shape[1] <= self.subArray:
				prob = torch.rand_like(weight)  # 生成 [0,1] 区间的随机数
				self.SA0_mask.append(torch.where(prob < self.SAF_prop))
				self.SA1_mask.append(torch.where(prob > 1 - self.SAF_prop))
			else:
				# need to divide to different subArray
				numSubArray = math.ceil(weight.shape[1]/self.subArray)
				padding_flag = numSubArray*self.subArray > weight.shape[1]
				if padding_flag:
					padding_num = numSubArray*self.subArray-weight.shape[1]
					weight = F.pad(weight,(0,padding_num),'constant',0)
					weight = weight.reshape(-1,numSubArray,self.subArray)
				else:
					weight = weight.reshape(-1,numSubArray,self.subArray)
				prob = torch.rand_like(weight)  # 生成 [0,1] 区间的随机数
				self.SA0_mask.append(torch.where(prob < self.SAF_prop))
				self.SA1_mask.append(torch.where(prob > 1 - self.SAF_prop))

	@torch.no_grad()
	def forward(self, input):
		inputOriginal_shape = input.shape
		# return F.linear(input, self.weight, self.bias)
		outputOrignal_shape = F.linear(input, self.weight, self.bias).shape
		output = torch.zeros(outputOrignal_shape).cuda().half()

		bitWeight = int(self.wl_weight)
		bitActivation = int(self.wl_input)

		# set parameters for Hardware Inference
		onoffratio = self.onoffratio
		upper = 1
		lower = 1/onoffratio
		cellRange = 2**self.cellBit
		if len(input.size())==3:
			input = input.reshape(-1,inputOriginal_shape[-1])

		if self.weight.shape[1] <= self.subArray:
			weight, w_scale = qw_tensor(self.weight, self.wl_weight, self.weight.shape[1])
			# quantize input into binary sequence
			inputQ, a_scale = qa_tensor(input, self.wl_input, per_tensor=True)
			for z in range(bitActivation-1):
				inputB = torch.fmod(inputQ, 2)
				inputQ = torch.round((inputQ-inputB)/2)
				X_decimal = weight
				outputD = torch.zeros(outputOrignal_shape).cuda().half()
				for k in range (int(bitWeight/self.cellBit)-1):
					remainder = torch.fmod(X_decimal, cellRange)
					X_decimal = torch.round((X_decimal-remainder)/cellRange)
					if self.vari_type == 'state_dependent':
						variation = torch.normal(0, self.vari, remainder.size()).cuda().half()
						remainderQ = remainder + remainder*variation 
					elif self.vari_type == 'laplace':
						laplace_dist = torch.distributions.Laplace(0, self.laplace_b)
						variation = laplace_dist.sample(remainder.size()).cuda().half()
						remainderQ = remainder + remainder*variation
					elif self.vari_type == 'state_independent':
						ind_variation = torch.normal(0, self.ind_vari, remainder.size()).cuda().half()
						remainderQ = remainder + ind_variation
					else:
						import ipdb; ipdb.set_trace()
					remainderQ[self.SA0_mask[k]] = 0
					remainderQ[self.SA1_mask[k]] = torch.sign(remainderQ[self.SA1_mask[k]])
					outputPartial= F.linear(inputB, remainderQ)
					# Add ADC quanization effects here !!!
					outputPartialQ = LinearQuantizeOut(outputPartial, self.ADCprecision)*w_scale.permute(1,0)
					scaler = cellRange**k
					outputD = outputD + (outputPartialQ*scaler).reshape(outputOrignal_shape)
				scalerIN = 2**z
				output += outputD * scalerIN * a_scale
		else:
			inputQ, a_scale = qa_tensor(input, self.wl_input, per_tensor=True)
			# need to divide to different subArray
			numSubArray = math.ceil(self.weight.shape[1]/self.subArray)
			padding_flag = numSubArray*self.subArray > self.weight.shape[1]
			if padding_flag:
				padding_num = numSubArray*self.subArray-self.weight.shape[1]
				inputQ = F.pad(inputQ,(0,padding_num),'constant',0).reshape(-1,numSubArray,self.subArray)
				weight = F.pad(self.weight,(0,padding_num),'constant',0)
				weight, w_scale = qw_tensor(weight, self.wl_weight, self.subArray)
				weight = weight.reshape(-1,numSubArray,self.subArray)
			else:
				weight, w_scale = qw_tensor(self.weight, self.wl_weight, self.subArray)
				weight = weight.reshape(-1,numSubArray,self.subArray)
				inputQ = inputQ.reshape(-1,numSubArray,self.subArray)
			for z in range(bitActivation-1):
				outputD = torch.zeros(outputOrignal_shape).cuda().half()

				inputB = torch.fmod(inputQ, 2)
				inputQ = torch.round((inputQ-inputB)/2)

				X_decimal = weight
				for k in range(int(bitWeight/self.cellBit)-1):
					remainder = torch.fmod(X_decimal, cellRange)
					X_decimal = torch.round((X_decimal-remainder)/cellRange)
					if self.vari_type == 'state_dependent':
						variation = torch.normal(0, self.vari, remainder.size()).cuda().half()
						remainderQ = remainder + remainder*variation 
					elif self.vari_type == 'laplace':
						laplace_dist = torch.distributions.Laplace(0, self.laplace_b)
						variation = laplace_dist.sample(remainder.size()).cuda().half()
						remainderQ = remainder + remainder*variation
					elif self.vari_type == 'state_independent':
						ind_variation = torch.normal(0, self.ind_vari, remainder.size()).cuda().half()
						remainderQ = remainder + ind_variation
					else:
						import ipdb; ipdb.set_trace()
					remainderQ[self.SA0_mask[k]] = lower
					remainderQ[self.SA1_mask[k]] = torch.sign(remainderQ[self.SA1_mask[k]])
					outputPartial = torch.matmul(inputB.permute(1,0,2),remainderQ.permute(1,2,0))
					# Add ADC quanization effects here !!!
					outputPartialQ = ((LinearQuantizeOut(outputPartial, self.ADCprecision)*w_scale.reshape(-1,numSubArray).permute(1,0)[:,None,:])).sum(0)
					scaler = cellRange**k
					outputD = outputD + (outputPartialQ*scaler).reshape(outputOrignal_shape)
				scalerIN = 2**z
				output += outputD * scalerIN * a_scale
		if self.bias is not None:
			output += self.bias[None,:]

		return output



def wrap_model(model, pim_params, exclude_layers=[]):
	wrapped_model = copy.deepcopy(model)
	modules_wrapper_list = []
	for name, module in wrapped_model.named_modules():
		if isinstance(module, nn.Conv2d) and module.groups == 1 and name not in exclude_layers:
			module_wrapper = QConv2d(name, module, pim_params)
			modules_wrapper_list.append(module_wrapper)
		elif isinstance(module, nn.Linear) and name not in exclude_layers:
			module_wrapper = QLinear(name, module, pim_params)
			modules_wrapper_list.append(module_wrapper)

	for module_wrapper in reversed(modules_wrapper_list):
		_setattr(wrapped_model, module_wrapper.name, module_wrapper)

	return wrapped_model



if __name__ == "__main__":
	pim_params={}
	pim_params['wl_weight'] = 8
	pim_params['wl_input'] = 8
	# Hardware Properties
	pim_params['subArray'] = 256           # size of subArray (e.g. 128*128)
	pim_params['ADCprecision'] = 6         # ADC precision (e.g. 5-bit)
	pim_params['cellBit'] = 1              # cell precision (e.g. 4-bit/cell)
	pim_params['onoffratio'] = 1e6         # device on/off ratio (e.g. Gmax/Gmin = 3)
	# if do not run the device retention / conductance variation effects, set args.vari=0, args.v=0
	pim_params['vari'] = 0                 # conductance variation (e.g. 0.1 standard deviation to generate random variation)

	
	
	# dummy_input = torch.randn(3,32,16,16).cuda().half()
	# module = nn.Conv2d(32,64,kernel_size=3).cuda().half()
	# wrap_module = QConv2d('test', module, pim_params)

	# output = module(dummy_input)
	# wrap_output = wrap_module(dummy_input)
	# print((wrap_output/output).mean())

	for i in range(100):
		dummy_input_1 = torch.randn(3,10,32).cuda().half()
		module = nn.Linear(32,64).cuda().half()
		wrap_module = QLinear('test', module, pim_params)

		output = module(dummy_input_1)
		wrap_output = wrap_module(dummy_input_1)
		print((wrap_output/output).mean())

		# # dummy_input_2 = torch.randn(30,32).cuda().half()
		# dummy_input_2 = dummy_input_1.reshape(-1,32)

		# output = module(dummy_input_2)
		# wrap_output = wrap_module(dummy_input_2)
		# print((wrap_output/output).mean())