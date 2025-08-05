#!/usr/bin/python
# -*-coding:utf-8-*-
import configparser as cp
import os
import math
from numpy import *
import numpy as np
import sys

class ReRAM_ProcessElement(object):
	def __init__(self, mode, pim_params, digital_freq, dim, pim_type):
		self.MODE = mode
		self.HW_config = pim_params
		self.DIM = (dim,dim) if isinstance(dim, int) else dim
		self.PIM_type = pim_type
		self.digital_frequency = digital_freq # unit: MHz
		self.digital_period = 1/self.digital_frequency*1e3 # unit: ns
		
		self.supply_voltage = 0.9 # unit: voltage(V)
		self.Tech = 28 # unit: nm

		self.RRAM_res = int(self.HW_config['RRAM_res'])
		self.calculate_xbar_size() # get xbar_size and array_num
		self.determin_adc_dac_config()

		self.PE_multiplex_xbar_num = [1,2]
		self.group_num = int(self.HW_config['wl_weight'] / self.RRAM_res) # default config: W4A8

		self.PE_group_ADC_num = self.xbar_size / self.ADC_reuse
		self.PE_group_DAC_num = self.xbar_size / self.DAC_reuse
		self.PE_ADC_num = self.group_num * self.PE_group_ADC_num * self.array_num
		self.PE_DAC_num = self.group_num * self.PE_group_DAC_num * self.array_num

		self.PE_xbar_num = self.group_num * self.PE_multiplex_xbar_num[0] * self.PE_multiplex_xbar_num[1] * self.array_num

		self.PE_area = 0
		self.PE_xbar_area = 0
		self.PE_ADC_area = 0
		self.PE_DAC_area = 0
		self.PE_adder_area = 0
		self.PE_shiftreg_area = 0
		self.PE_iReg_area = 0
		self.PE_oReg_area = 0
		self.PE_digital_area = 0

		self.PE_read_power = 0
		self.PE_xbar_read_power = 0
		self.PE_ADC_read_power = 0
		self.PE_DAC_read_power = 0
		self.PE_adder_read_power = 0
		self.PE_shiftreg_read_power = 0
		self.PE_iReg_read_power = 0
		self.PE_oReg_read_power = 0
		self.PE_digital_read_power = 0

		self.PE_write_power = 0
		self.PE_xbar_write_power = 0
		self.PE_ADC_write_power = 0
		self.PE_DAC_write_power = 0
		self.PE_adder_write_power = 0
		self.PE_shiftreg_write_power = 0
		self.PE_iReg_write_power = 0
		self.PE_oReg_write_power = 0
		self.PE_digital_write_power = 0

		self.PE_read_latency = 0
		self.PE_xbar_read_latency = 0
		self.PE_ADC_read_latency = 0
		self.PE_DAC_read_latency = 0
		self.PE_adder_read_latency = 0
		self.PE_shiftreg_read_latency = 0
		self.PE_iReg_read_latency = 0
		self.PE_oReg_read_latency = 0
		self.PE_digital_read_latency = 0

		self.PE_write_latency = 0
		self.PE_xbar_write_latency = 0
		self.PE_ADC_write_latency = 0
		self.PE_DAC_write_latency = 0
		self.PE_adder_write_latency = 0
		self.PE_shiftreg_write_latency = 0
		self.PE_iReg_write_latency = 0
		self.PE_oReg_write_latency = 0
		self.PE_digital_write_latency = 0

		self.PE_read_energy = 0
		self.PE_xbar_read_energy = 0
		self.PE_ADC_read_energy = 0
		self.PE_DAC_read_energy = 0
		self.PE_adder_read_energy = 0
		self.PE_shiftreg_read_energy = 0
		self.PE_iReg_read_energy = 0
		self.PE_oReg_read_energy = 0
		self.PE_digital_read_energy = 0

		self.PE_write_energy = 0
		self.PE_xbar_write_energy = 0
		self.PE_ADC_write_energy = 0
		self.PE_DAC_write_energy = 0
		self.PE_adder_write_energy = 0
		self.PE_shiftreg_write_energy = 0
		self.PE_iReg_write_energy = 0
		self.PE_oReg_write_energy = 0
		self.PE_digital_write_energy = 0

		self.calculate_PE_adder_num()

	def calculate_xbar_size(self):
		if self.MODE == 'static':
			self.xbar_size = int(self.HW_config['XBAR_size'])
		else:
			self.xbar_size = int(self.HW_config['XBAR_size'])
		self.array_num = math.ceil(self.DIM[0]/self.xbar_size) * math.ceil(self.DIM[1]/self.xbar_size)
	
	def determin_adc_dac_config(self):
		self.DAC_res = 1
		self.DAC_reuse = int(self.HW_config['DAC_reuse'])
		self.ADC_reuse = int(self.HW_config['ADC_reuse'])
		self.ADC_res = int(self.HW_config['ADC_resolution'])


	def calculate_PE_adder_num(self):
		self.PE_adder_num = 0
		for i in range(1,math.ceil(math.log2(self.group_num))):
			self.PE_adder_num += (self.ADC_res + i - 1) *  self.group_num / 2**i
	
	def calculate_xbar_area(self):
		# unit: um^2
		# use minimal device area in published papers, consider 1T1R structure and not consider driver area
		# 1T1R is 6F^2 and 0T1R is 4F^2
		minimal_device_area = 1e-2 * 6 / 4
		self.xbar_area = self.xbar_size**2 * minimal_device_area

		return self.xbar_area
	
	def calculate_DAC_area(self):
		# unit: um^2
		# reference: 
		# 1. A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
		# 2. Analysis of Power Consumption and Linearity in Capacitive Digital-to-Analog Converters Used in Successive Approximation ADCs
		k = 0.064
		self.DAC_area = k * (2**self.DAC_res) * (self.Tech/28)**2
		return self.DAC_area

	def calculate_ADC_area(self):
		# unit: um^2
		# reference: ADC Performance Survey 1997-2024
		self.ADC_area = 10**(-0.25*self.ADC_res+5.2) * 2**(self.ADC_res) * (self.Tech*1e-3)**2
		return self.ADC_area
	
	def calculate_adder_area(self):
		# reference: Analog or Digital In-memory Computing? Benchmarking through Quantitative Modeling
		# unit: um^2
		self.adder_area = 7.8 * 0.614 * (self.Tech/28)**2 # 1-bit Full-Adder
		return self.adder_area

	def calculate_shiftreg_area(self):
		# unit: um^2
		self.shiftreg_area = 16.38 * (self.Tech/28)**2
		return self.shiftreg_area
	
	def calculate_iReg_area(self):
		# unit: um^2
		# reference: Analog or Digital In-memory Computing? Benchmarking through Quantitative Modeling
		self.iReg_area = 6 * 0.614 * (self.Tech/28)**2
		return self.iReg_area
	
	def calculate_oReg_area(self):
		# unit: um^2
		# reference: Analog or Digital In-memory Computing? Benchmarking through Quantitative Modeling
		self.oReg_area = 6 * 0.614 * (self.Tech/28)**2
		return self.oReg_area
	
	def calculate_area(self, SimConfig_path=None):
		# unit: um^2
		self.calculate_xbar_area()
		self.calculate_DAC_area()
		self.calculate_ADC_area()
		self.calculate_adder_area()
		self.calculate_shiftreg_area()
		self.calculate_iReg_area()
		self.calculate_oReg_area()
		self.PE_xbar_area = self.PE_xbar_num*self.xbar_area
		self.PE_ADC_area = self.PE_ADC_num*self.ADC_area
		self.PE_DAC_area = self.PE_DAC_num*self.DAC_area
		self.PE_adder_area = self.PE_group_ADC_num*self.PE_adder_num*self.adder_area
		self.PE_shiftreg_area = self.PE_ADC_num*self.shiftreg_area
		self.PE_iReg_area = self.PE_DAC_num*self.iReg_area # 1-bit input
		self.PE_oReg_area = self.PE_group_ADC_num*(math.ceil(math.log2(self.group_num))+self.ADC_res)*self.oReg_area # multiple-bit outout
		self.PE_digital_area = self.PE_adder_area + self.PE_shiftreg_area + self.PE_iReg_area + self.PE_oReg_area
		self.PE_area = self.PE_xbar_area + self.PE_ADC_area + self.PE_DAC_area + self.PE_digital_area

		return self.PE_area

	def calculate_xbar_read_power(self):
		# unit: W
		# use minimal device power in published papers, consider 1T1R structure and not consider driver area
		minimal_device_read_power = 1.025e-13
		self.xbar_read_power = self.xbar_size**2 * minimal_device_read_power

		return self.xbar_read_power
	
	def calculate_DAC_power(self):
		# reference: 
		# 1. A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
		# 2. Analysis of Power Consumption and Linearity in Capacitive Digital-to-Analog Converters Used in Successive Approximation ADCs
		# unit: W
		k = 2.407*1e-6
		self.DAC_power = k * (2**self.DAC_res) * (self.supply_voltage**2)
		return self.DAC_power

	def calculate_ADC_power(self):
		# reference: ADC Performance Survey 1997-2024
		# unit: W
		self.calculate_ADC_latency()
		self.ADC_energy = 1e-3*(1.44+0.121*1e-3*(4**self.ADC_res)*(self.supply_voltage**2)) # unit: nJ
		self.ADC_power = self.ADC_energy / self.ADC_latency
		return self.ADC_power
	
	def calculate_adder_power(self):
		# reference: Analog or Digital In-memory Computing? Benchmarking through Quantitative Modeling
		# unit: W
		self.adder_energy = 6 * 0.7 * self.supply_voltage**2 * 1e-6
		self.adder_power = self.adder_energy / self.PE_adder_latency
		return self.adder_power

	def calculate_shiftreg_power(self):
		# W
		self.shiftreg_power = 2.83 * 1e-6
		return self.shiftreg_power
	
	def calculate_iReg_power(self):
		# unit: W
		self.iReg_power = 3*0.7*(self.supply_voltage**2)*1e-6 / self.digital_period
		return self.iReg_power
	
	def calculate_oReg_power(self):
		# unit: W
		self.oReg_power = 3*0.7*(self.supply_voltage**2)*1e-6 / self.digital_period
		return self.oReg_power

	def calculate_read_power(self):
		# unit: W
		self.calculate_xbar_read_power()
		self.calculate_DAC_power()
		self.calculate_ADC_power()
		self.calculate_shiftreg_power()
		self.calculate_iReg_power()
		self.calculate_oReg_power()
		self.calculate_adder_power()
		self.PE_read_power = 0
		self.PE_xbar_read_power = 0
		self.PE_ADC_read_power = 0
		self.PE_DAC_read_power = 0
		self.PE_adder_read_power = 0
		self.PE_shiftreg_read_power = 0
		self.PE_iReg_read_power = 0
		self.PE_oReg_read_power = 0
		self.PE_digital_read_power = 0

		self.PE_xbar_read_power = self.PE_multiplex_xbar_num[1] * self.group_num * self.xbar_read_power / self.DAC_reuse / self.ADC_reuse
		self.PE_DAC_read_power = self.PE_DAC_num * self.DAC_power
		self.PE_iReg_read_power = self.PE_DAC_num * self.iReg_power

		if self.PIM_type == 'RRAM': # analog pim
			self.PE_ADC_read_power = self.group_num * math.ceil(self.xbar_size / self.ADC_reuse) * self.ADC_power
			self.PE_adder_read_power = (self.group_num - 1) * math.ceil(self.xbar_size / self.ADC_reuse) * self.adder_power
			self.PE_shiftreg_read_power = self.group_num * math.ceil(self.xbar_size / self.ADC_reuse) * self.shiftreg_power
			self.PE_oReg_read_power = self.PE_group_ADC_num*(math.ceil(math.log2(self.group_num))+self.ADC_res) * self.oReg_power
		else: # digital pim
			# TODO: add digital pim
			pass

		self.PE_digital_read_power = self.PE_adder_read_power + self.PE_shiftreg_read_power + self.PE_iReg_read_power + self.PE_oReg_read_power
		self.PE_read_power = self.PE_xbar_read_power + self.PE_DAC_read_power + self.PE_ADC_read_power + self.PE_digital_read_power

		return self.PE_read_power

	def calculate_xbar_write_power(self):
		# unit: W
		# use minimal device power in published papers, consider 1T1R structure and not consider driver area
		minimal_device_write_power = 2.56*1e-12
		self.xbar_write_power = self.xbar_size**2 * minimal_device_write_power

		return self.xbar_write_power

	def calculate_write_power(self, op):
		# unit: W

		self.calculate_read_power()
		self.calculate_xbar_write_power()
		self.calculate_DAC_power()
		self.calculate_iReg_power()
		self.PE_write_power = 0
		self.PE_xbar_write_power = 0
		self.PE_DAC_write_power = 0
		self.PE_iReg_write_power = 0
		self.PE_digital_write_power = 0

		if op in ['EWM','RMSNorm','LayerNorm', 'SiLU']:
			write_column = 1
		elif op in ['MVM_dynamic']:
			write_column = self.xbar_size
		self.PE_xbar_write_power = self.PE_multiplex_xbar_num[1]*self.group_num*self.xbar_write_power*write_column/self.DAC_reuse/self.xbar_size
		self.PE_DAC_write_power = self.group_num*math.ceil(self.xbar_size/self.DAC_reuse)*self.DAC_power
		self.PE_iReg_write_power = self.group_num*math.ceil(self.xbar_size/self.DAC_reuse)*self.iReg_power

		self.PE_digital_write_power = self.PE_iReg_write_power
		self.PE_write_power = self.PE_xbar_write_power + self.PE_DAC_write_power + self.PE_digital_write_power

		# consider read-based verify process
		self.PE_verify_power = self.PE_read_power/self.xbar_size # verify one row each cycle
		self.PE_write_power += self.PE_verify_power

		return self.PE_write_power

	def calculate_xbar_read_latency(self):
		# unit: ns
		# use minimal device power in published papers, consider 1T1R structure and not consider driver area
		# TODO: find a suitable value in publised papers
		self.xbar_read_latency = 1

		return self.xbar_read_latency
	
	def calculate_DAC_latency(self):
		# unit: ns
		self.DAC_sample_rate = self.digital_frequency*1e-3 # unit: GHz
		self.DAC_latency = 1 / self.DAC_sample_rate * (self.DAC_res + 2)
		return self.DAC_latency

	def calculate_ADC_latency(self):
		# reference: ADC Performance Survey 1997-2024
		# unit: ns
		self.ADC_latency = 0.036 + 0.115*1e-3*(4**self.ADC_res)
		return self.ADC_latency
	
	def calculate_adder_latency(self):
		# reference: ADC Performance Survey 1997-2024
		# unit: ns
		self.adder_latency = 4.8*47.8*1e-3*math.ceil(math.log2(self.group_num))+2*47.8*1e-3*(math.ceil(math.log2(self.group_num))+self.ADC_res)
		return self.adder_latency

	def calculate_read_latency(self):
		multiple_time = math.ceil(8/self.DAC_res) * math.ceil(self.xbar_size/self.PE_group_DAC_num) * \
						math.ceil(self.xbar_size/self.PE_group_ADC_num)

		self.calculate_DAC_latency()
		self.calculate_ADC_latency()
		self.calculate_xbar_read_latency()
		self.calculate_adder_latency()
		self.PE_xbar_read_latency = multiple_time * self.xbar_read_latency
		self.PE_DAC_read_latency = multiple_time * self.DAC_latency
		self.PE_ADC_read_latency = multiple_time * self.ADC_latency
		self.PE_iReg_latency = multiple_time * self.digital_period
		self.PE_oReg_latency = self.digital_period
		self.PE_shiftreg_latency =  multiple_time * self.digital_period
		# self.PE_adder_latency = math.ceil(math.log2(self.group_num))*self.digital_period
		# reference: Analog or Digital In-memory Computing? Benchmarking through Quantitative Modeling
		self.PE_adder_latency =  multiple_time * self.adder_latency
		self.PE_digital_read_latency = self.PE_iReg_latency+self.PE_shiftreg_latency+\
									   self.PE_adder_latency+self.PE_oReg_latency

		self.PE_read_latency = self.PE_xbar_read_latency+self.PE_DAC_read_latency+self.PE_ADC_read_latency+self.PE_digital_read_latency

		return self.PE_read_latency

	def calculate_xbar_write_latency(self):
		# unit: ns
		# use minimal device power in published papers, consider 1T1R structure and not consider driver area
		# reference: Ultra-fast switching memristors based on two-dimensional materials / per write cycle latency
		self.xbar_write_latency = 2.7

		return self.xbar_write_latency
	
	def calculate_write_latency(self, op):
		# 4-bit weight / 8-bit activation write-verify latency
		# multiple_time = math.ceil(self.xbar_size/self.PE_group_DAC_num)
		# W-V optimization reference:
		# [1] Write or Not: Programming Scheme Optimization for RRAM-based Neuromorphic Computing
		# [2] An Efficient Programming Framework for Memristor-based Neuromorphic Computing 
		multiple_time = self.xbar_size * 20 * 0.1 # 20 upper bound cycles, 0.1 optimization space
		
		self.calculate_DAC_latency()
		self.calculate_xbar_write_latency()
		self.calculate_read_latency()
		self.PE_xbar_write_latency = multiple_time * self.xbar_write_latency
		self.PE_DAC_write_latency = multiple_time * self.DAC_latency
		self.PE_iReg_write_latency = multiple_time*self.digital_period
		self.PE_digital_write_latency = self.PE_iReg_write_latency

		self.PE_write_latency = self.PE_xbar_write_latency+self.PE_DAC_write_latency+self.PE_digital_write_latency
		# consider read-based verify process
		self.PE_verify_latency = multiple_time * self.PE_read_latency / math.ceil(8/self.DAC_res) / math.ceil(self.xbar_size/self.PE_group_DAC_num) # verify one row each cycle
		self.PE_write_latency += self.PE_verify_latency


		return self.PE_write_latency

	def calculate_read_energy(self):
		self.calculate_read_power()
		self.calculate_read_latency()

		self.PE_xbar_read_energy = self.PE_xbar_read_latency * self.PE_xbar_read_power
		self.PE_DAC_read_energy = self.PE_DAC_read_latency * self.PE_DAC_read_power
		self.PE_ADC_read_energy = self.PE_ADC_read_latency * self.PE_ADC_read_power
		self.PE_iReg_read_energy = self.PE_iReg_latency * self.PE_iReg_read_power
		self.PE_shiftreg_read_energy = self.PE_shiftreg_latency * self.PE_shiftreg_read_power
		self.PE_adder_read_energy = self.PE_adder_latency * self.PE_adder_read_power
		self.PE_oReg_read_energy = self.PE_oReg_latency * self.PE_oReg_read_power
		self.PE_digital_read_energy = self.PE_iReg_read_energy+self.PE_shiftreg_read_energy+\
									  self.PE_adder_read_energy+self.PE_oReg_read_energy

		self.PE_read_energy = self.PE_xbar_read_energy+self.PE_DAC_read_energy+self.PE_ADC_read_energy+self.PE_digital_read_energy

		return self.PE_read_energy
	
	def calculate_write_energy(self,op):
		self.calculate_read_energy()
		self.calculate_write_power(op)
		self.calculate_write_latency(op)

		self.PE_xbar_write_energy = self.PE_xbar_write_latency * self.PE_xbar_write_power
		self.PE_DAC_write_energy = self.PE_DAC_write_latency * self.PE_DAC_write_power
		self.PE_iReg_write_energy = self.PE_iReg_write_latency * self.PE_iReg_write_power
		self.PE_digital_write_energy = self.PE_iReg_write_energy+self.PE_shiftreg_write_energy

		self.PE_write_energy = self.PE_xbar_write_energy+self.PE_DAC_write_energy+self.PE_digital_write_energy
		self.PE_verify_energy = self.PE_read_energy / math.ceil(8/self.DAC_res)

		# consider read-based verify process
		self.PE_write_energy += self.PE_verify_energy

		return self.PE_write_energy


	def PE_output(self):
		print("---------------------Crossbar Configurations-----------------------")
		crossbar.xbar_output(self)
		print("-------------------------PE Configurations-------------------------")
		print("total crossbar number in one PE:", self.PE_xbar_num)
		print("			the number of crossbars sharing a set of interfaces:",self.PE_multiplex_xbar_num)
		print("total DAC number in one PE:", self.PE_DAC_num)
		print("			the number of DAC in one set of interfaces:", self.PE_group_DAC_num)
		print("total ADC number in one PE:", self.PE_ADC_num)
		print("			the number of ADC in one set of interfaces:", self.PE_group_ADC_num)
		print("---------------------PE Area Simulation Results--------------------")
		print("PE area:", self.PE_area, "um^2")
		print("			crossbar area:", self.PE_xbar_area, "um^2")
		print("			DAC area:", self.PE_DAC_area, "um^2")
		print("			ADC area:", self.PE_ADC_area, "um^2")
		print("			digital part area:", self.PE_digital_area, "um^2")
		print("			|---adder area:", self.PE_adder_area, "um^2")
		print("			|---shift-reg area:", self.PE_shiftreg_area, "um^2")
		print("--------------------PE Latency Simulation Results-----------------")
		print("PE read latency:", self.PE_read_latency, "ns")
		print("			crossbar read latency:", self.PE_xbar_read_latency, "ns")
		print("			DAC read latency:", self.PE_DAC_read_latency, "ns")
		print("			ADC read latency:", self.PE_ADC_read_latency, "ns")
		print("			digital part read latency:", self.PE_digital_read_latency, "ns")
		print("			|---adder read latency:", self.PE_adder_latency, "ns")
		print("			|---shift-reg read latency:", self.PE_shiftreg_latency, "ns")
		print("PE write latency:", self.PE_write_latency, "ns")
		print("			crossbar write latency:", self.PE_xbar_write_latency, "ns")
		print("			DAC write latency:", self.PE_DAC_write_latency, "ns")
		print("			digital part write latency:", self.PE_digital_write_latency, "ns")
		print("--------------------PE Power Simulation Results-------------------")
		print("PE read power:", self.PE_read_power, "W")
		print("			crossbar read power:", self.PE_xbar_read_power, "W")
		print("			DAC read power:", self.PE_DAC_read_power, "W")
		print("			ADC read power:", self.PE_ADC_read_power, "W")
		print("			digital part read power:", self.PE_digital_read_power, "W")
		print("			|---adder power:", self.PE_adder_read_power, "W")
		print("			|---shift-reg power:", self.PE_shiftreg_read_power, "W")
		print("PE write power:", self.PE_write_power, "W")
		print("			crossbar write power:", self.PE_xbar_write_power, "W")
		print("			DAC write power:", self.PE_DAC_write_power, "W")
		print("			ADC write power:", self.PE_ADC_write_power, "W")
		print("			digital part write power:", self.PE_digital_write_power, "W")
		print("------------------PE Energy Simulation Results--------------------")
		print("PE read energy:", self.PE_read_energy, "nJ")
		print("			crossbar read energy:", self.PE_xbar_read_energy, "nJ")
		print("			DAC read energy:", self.PE_DAC_read_energy, "nJ")
		print("			ADC read energy:", self.PE_ADC_read_energy, "nJ")
		print("			digital part read energy:", self.PE_digital_read_energy, "nJ")
		print("PE write energy:", self.PE_write_energy, "nJ")
		print("			crossbar write energy:", self.PE_xbar_write_energy, "nJ")
		print("			DAC write energy:", self.PE_DAC_write_energy, "nJ")
		print("			ADC write energy:", self.PE_ADC_write_energy, "nJ")
		print("			digital part write energy:", self.PE_digital_write_energy, "nJ")
		print("-----------------------------------------------------------------")
	
def PE_test():
	print("load file:",test_SimConfig_path)
	_PE = ProcessElement(test_SimConfig_path)
	_PE.calculate_area(test_SimConfig_path)
	_PE.calculate_read_power(_PE.xbar_size[1], _PE.xbar_size[0], _PE.group_num, test_SimConfig_path)
	_PE.calculate_write_power(_PE.xbar_size[1], _PE.xbar_size[0], _PE.group_num, test_SimConfig_path)
	_PE.calculate_energy_efficiency(test_SimConfig_path)
	_PE.calculate_write_energy_efficiency(test_SimConfig_path)
	_PE.PE_output()


if __name__ == '__main__':
	PE_test()