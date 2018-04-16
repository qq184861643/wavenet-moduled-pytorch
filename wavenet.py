import torch
from torch import nn
import torch.nn.functional as F
from layer_utils import ResidualBlock,CausalConv1d
from utils import *

class WaveNet(nn.Module):
	def __init__(self,in_depth=256,dilation_channels=32,
					res_channels=32,skip_channels=512,end_channels=256,
					kernel_size=2,bias=False,dilation_depth=10,n_blocks=5):
		super(WaveNet,self).__init__()
		self.n_blocks = n_blocks
		self.dilation_depth = dilation_depth

		self.pre = nn.Embedding(in_depth,res_channels)
		self.pre_conv = CausalConv1d(res_channels,res_channels,kernel_size,bias=bias)

		self.dilations = []
		self.resblocks = nn.ModuleList()
		init_dilation=1
		receptive_field = 2
		for i in range(n_blocks):
			addition_scope = kernel_size-1
			new_dilation = 1
			for i in range(dilation_depth):
				self.dilations.append((new_dilation,init_dilation))
				self.resblocks.append(ResidualBlock(dilation_channels,res_channels,
														skip_channels,kernel_size,bias))
				receptive_field+=addition_scope
				addition_scope*=2
				init_dilation = new_dilation
				new_dilation*=2


		self.post = nn.Sequential(nn.ReLU(),
								  nn.Conv1d(skip_channels,end_channels,1,bias=True),
								  nn.ReLU(),
								  nn.Conv1d(end_channels,in_depth,1,bias=True))
		self.receptive_field = receptive_field
	def forward(self,inputs):
		x = self.preprocess(inputs)
		#print x.size()
		skip = 0

		for i in range(self.n_blocks*self.dilation_depth):
			(dilation,init_dilation) = self.dilations[i]
			x,s = self.resblocks[i](x,dilation,init_dilation)
			try:
				skip = skip[:,:,-s.size(2):]
			except:
				skip = 0
			#if not isinstance(skip,int):
				#print 'skip',skip.size(),'s',s.size()
			skip = skip+s

		outputs = self.post(skip)

		return outputs

	def preprocess(self,inputs):
		#print 'before',inputs.size()
		out = self.pre(inputs).transpose(1,2)
		#print 'before',out.size()
		out = self.pre_conv(out)
		#print 'after',out.size()
		return out