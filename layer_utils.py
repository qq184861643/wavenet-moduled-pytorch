import torch
from torch import nn
import torch.nn.functional as F
from utils import dilate

class CausalConv1d(nn.Module):
	def __init__(self,in_channels,out_channels,kernel_size=2,stride=1,
					dilation=1,bias=True):
		super(CausalConv1d,self).__init__()

		self.pad = (kernel_size - 1) * dilation
		self.conv = nn.Conv1d(in_channels,out_channels,kernel_size,
							stride=stride,padding=self.pad,dilation=dilation,bias=bias)

	def forward(self,inputs):
		
		outputs = self.conv(inputs)
		return outputs[:,:,:-self.pad]


class ResidualBlock(nn.Module):
	def __init__(self,dilation_channels,res_channels,skip_channels,kernel_size,bias):
		super(ResidualBlock,self).__init__()
		self.filter_conv = nn.Conv1d(in_channels=res_channels,out_channels=dilation_channels,kernel_size=kernel_size,bias=bias)
		self.gate_conv = nn.Conv1d(in_channels = res_channels,out_channels = dilation_channels,kernel_size=kernel_size,bias=bias)
		self.skip_conv = nn.Conv1d(in_channels = dilation_channels,out_channels = skip_channels,kernel_size=1,bias=bias)
		self.res_conv = nn.Conv1d(in_channels = dilation_channels,out_channels = res_channels,kernel_size=1,bias=bias)
		self.kernel_size = kernel_size
	def forward(self,inputs,dilation,init_dilation):
		inputs = dilate(inputs,dilation,init_dilation)
		sigmoid_out = F.sigmoid(self.gate_conv(inputs))
		tanh_out = F.tanh(self.filter_conv(inputs)) 
		hidden = sigmoid_out*tanh_out
		skip = hidden
		if hidden.size(2)!=1:
			skip = dilate(hidden,1,init_dilation=dilation)
		skip_out = self.skip_conv(skip)
		#print 'hidden',hidden.size(),'s',skip.size()
		res_out = self.res_conv(hidden)
		#print res_out.size()
		#print inputs[:,:,-res_out.size(2):].size()
		outputs = res_out+inputs[:,:,(self.kernel_size-1):]
		return outputs,skip_out

