#     Copyright 2020 Google LLC
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#         https://www.apache.org/licenses/LICENSE-2.0
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.


import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

DELTA = 1e-10


# For collating data
def collate(examples, pad_token_id):
	return pad_sequence(examples, batch_first=True, padding_value=pad_token_id)


def del_attr(obj, names):
	if len(names) == 1:
		delattr(obj, names[0])
	else:
		del_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
	if len(names) == 1:
		setattr(obj, names[0], val)
	else:
		set_attr(getattr(obj, names[0]), names[1:], val)

# Converts a list of tensors to a vector
def vectorize(list_of_tensors):
	orig_shapes, vec = [], []
	with torch.no_grad():
		for tensor in list_of_tensors:
			orig_shapes.append(tensor.shape)
			vec.append(tensor.view(-1))
		vec = torch.cat(vec)
	return orig_shapes, vec


# My personal implementation of Gram-Schmidt orthogonalization
def my_qr(tensor):
	for i in range(tensor.shape[0]):
		if i == 0:
			tensor[i].div_(tensor[i].norm() + DELTA)
		else:
			proj_ = (tensor[i].unsqueeze(0)).matmul(tensor[:i].t()).matmul(tensor[:i])
			tensor[i] = tensor[i] - proj_
			tensor[i].div_(tensor[i].norm() + DELTA)
	return tensor


# Convert a list of list of tensors into a matrix.
def batch_vectorize(list_of_list_of_tensors):
	mat = []
	with torch.no_grad():
		for p_idx in range(len(list_of_list_of_tensors[0])):
			this_vec = []
			for list_of_tensor in list_of_list_of_tensors:
				this_vec.append(list_of_tensor[p_idx].view(-1))
			mat.append(torch.stack(this_vec))
	return torch.cat(mat, dim=1)

# Takes in a gradient vector and the shapes of parameter gradients.
# Converts the gradient vector into a list of per-parameter gradient matrices
def reshape_grad_vector(new_grad_vec, grad_shapes):
	new_grads, cur_pos = [], 0
	for shape in grad_shapes:
		delta_shape = np.prod(shape)
		sub_vec = new_grad_vec[cur_pos: (cur_pos + delta_shape)]
		new_grads.append(sub_vec.reshape(shape))
		cur_pos += delta_shape
	return new_grads


# Project from low_rank back to original gradient dimention
def get_new_grads(low_rank, V, grad_shapes, stats):
	new_grad_vec = low_rank.matmul(V).squeeze()
	new_grad_vec = (new_grad_vec * stats[1]) + stats[0]
	new_norm = new_grad_vec.norm()
	return new_grad_vec, new_norm


# To reset the dropout rate in the model.
# Useful when we wish dropout to be in eval mode but rest of model to be in train-mode
def set_model_dropout(model, dp):
	old_droprate = None
	if hasattr(model, 'dp_ratio'):
		old_droprate = model.dp_ratio
		model.dp_ratio = dp
	else:
		for name, child in model.named_modules():
			if hasattr(child, 'droprate'):
				old_droprate = child.droprate
				child.droprate = dp
	return old_droprate


'''
This code is borrowed and modified from :
https://github.com/pytorch/contrib/blob/master/torchcontrib/optim/swa.py
'''
def bn_update(loader, model, multitask=False, device=None, group=None):
	"""Updates BatchNorm running_mean, running_var buffers in the model.
		It performs one pass over data in `loader` to estimate the activation
		statistics for BatchNorm layers in the model.
		Args:
			loader (torch.utils.data.DataLoader): dataset loader to compute the
				activation statistics on. Each data batch should be either a
				tensor, or a list/tuple whose first element is a tensor
				containing data.
			model (torch.nn.Module): model for which we seek to update BatchNorm
				statistics.
			device (torch.device, optional): If set, data will be trasferred to
				:attr:`device` before being passed into :attr:`model`.
	"""
	if not _check_bn(model):
		return
	was_training = model.training
	model.train()
	momenta = {}
	model.apply(_reset_bn)
	model.apply(lambda module: _get_momenta(module, momenta))
	n = 0
	for input in loader:
		if multitask:
			b = input[0][0].size()[0] + input[0][1].size()[0]
		else:
			b = input[0].size(0)
		momentum = b / float(n + b)
		for module in momenta.keys():
			module.momentum = momentum

		if device is not None:
			input = input.to(device)
		if multitask:
			x1, x2 = input[0][0], input[0][1]
			x = torch.cat([x1, x2], dim=0)
		else:
			x, _ = input
		model(x, head_name=group)
		n += b
	model.apply(lambda module: _set_momenta(module, momenta))
	model.train(was_training)


# BatchNorm utils
def _check_bn_apply(module, flag):
	if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
		flag[0] = True


def _check_bn(model):
	flag = [False]
	model.apply(lambda module: _check_bn_apply(module, flag))
	return flag[0]


def _reset_bn(module):
	if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
		module.running_mean = torch.zeros_like(module.running_mean)
		module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
	if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
		momenta[module] = module.momentum


def _set_momenta(module, momenta):
	if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
		module.momentum = momenta[module]
