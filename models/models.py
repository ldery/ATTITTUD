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

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, kaiming_uniform_, zeros_, kaiming_normal_
import torch
import pdb
from torchvision import models
from .wideresnet import WideResNet


def add_model_opts(parser):
	parser.add_argument('-init-method', type=str, default='xavier_unif', choices=['xavier_unif', 'kaiming_unif'])
	parser.add_argument('-loss-fn', type=str, default='CE', choices=['CE', 'BCE', 'MSE', 'BCEWithLogitsLoss'])
	parser.add_argument('-model-type', type=str, choices=['MLP', 'WideResnet', 'SimpleCNN', 'ResNet'], default='MLP')
	parser.add_argument('-base-resnet', type=str, choices=['18', '50'], default='18')
	parser.add_argument('-pretrained', action='store_true')
	# For the Linear Model
	parser.add_argument('-layer-sizes', type=str, default='[784, 100, 10]')
	# For WideResnet Model
	parser.add_argument('-depth', type=int, default=22)
	parser.add_argument('-widen-factor', type=int, default=4)
	parser.add_argument('-dropRate', type=float, default=0.1)
	parser.add_argument('-ft-dropRate', type=float, default=0.1)
	return parser


# Recursively delete attribute and its children
def del_attr(obj, names):
	if len(names) == 1:
		delattr(obj, names[0])
	else:
		del_attr(getattr(obj, names[0]), names[1:])


# Recursively set attribute and its children
def set_attr(obj, names, val):
	if len(names) == 1:
		setattr(obj, names[0], val)
	else:
		set_attr(getattr(obj, names[0]), names[1:], val)


# Weight initialization
def weight_init(init_method):

	def initfn(layer):
		if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
			if init_method == 'xavier_unif':
				xavier_uniform_(layer.weight.data)
			elif init_method == 'kaiming_unif':
				kaiming_uniform_(layer.weight.data)
			elif init_method == 'kaiming_normal':
				kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
			if layer.bias is not None:
				zeros_(layer.bias.data)
		elif isinstance(layer, nn.BatchNorm2d):
			layer.weight.data.fill_(1)
			layer.bias.data.zero_()
	return initfn


# Super-class encapsulating all model related functions
class Model(nn.Module):
	def __init__(
					self, loss_name='CE', make_funct=False,
					pca_layers=None, **kwargs
				):
		super(Model, self).__init__()
		# make the model functional so we can use jacobian-vector-product
		self.make_funct = make_funct
		self.loss_fn_name = loss_name
		self.src_loss = self.get_loss_fn(loss_name)
		# the layers whose gradient we want to do a lowrank approx of
		self.pca_layers = pca_layers
		self.is_functional = make_funct
		self.params, self.names = [], []
		self.loss_fn = self.get_loss_fn(loss_name)

	def get_loss_fn(self, fn_name, reduction='mean'):
		if fn_name == 'CE':
			return nn.CrossEntropyLoss(reduction=reduction)
		elif fn_name == 'BCE':
			return nn.BCELoss(reduction=reduction)
		elif fn_name == 'MSE':
			return nn.MSELoss(reduction=reduction)
		elif fn_name == 'BCEWLogits':
			return nn.BCEWithLogitsLoss(reduction=reduction)

	def param_shapes(self):
		if hasattr(self, 'p_shapes'):
			return self.p_shapes
		self.p_shapes = [x.shape for x in self.parameters()]
		return self.p_shapes

	# Model needs to be in functional mode for us to compute JVP
	def make_functional(self):
		already_stripped = False
		if len(self.params):
			already_stripped = True
			name_and_params = zip(self.names, self.params)
		else:
			name_and_params = list(self.named_parameters())
		for name, p in name_and_params:
			# strip the model of these attributes so it's purely functional
			del_attr(self, name.split("."))
			if not already_stripped:
				self.names.append(name)
				self.params.append(p)
		if not already_stripped:
			self.params = tuple(self.params)
		return self.params, self.names

	def parameters(self):
		if self.is_functional and len(self.params):
			return self.params
		else:
			return super(Model, self).parameters()

	def named_parameters(self, recurse=True):
		if self.is_functional and len(self.names):
			return zip(self.names, self.params)
		else:
			return super(Model, self).named_parameters(recurse=recurse)

	def functional_(self, inp_, head_name=None):
		'''
			head_name (str) : the name of the model heads to apply as final layer to this input.
			Necessary because we are doing multiheaded multitasking
		'''
		loss_fn = self.get_loss_fn(self.loss_fn_name, reduction='none')

		def fn(*params):
			for name, p in zip(self.names, params):
				set_attr(self, name.split("."), p)
			if isinstance(inp_, tuple) or isinstance(inp_, list):
				x, y = inp_
				result = loss_fn(self.model(x, head_name=head_name), y)
				if hasattr(self, 'is_chexnet') and head_name == 'tgt':
					result = result.mean(dim=-1)
				return result
			else:
				return self.model(inp_, head_name=head_name)
		return fn

	def forward(self, x, head_name=None):
		if self.is_functional:
			m_out = self.functional_(x, head_name=head_name)(*self.params)
		else:
			m_out = self.model(x, head_name=head_name)
		if self.loss_fn_name == 'BCE':
			# We need to do a sigmoid if we're using binary labels
			m_out = torch.sigmoid(m_out)
		return m_out

	def criterion(self, outs, target, head_name='tgt'):
		if head_name == 'src':
			assert hasattr(self, 'src_loss'), 'src loss requested but no function found'
			return self.src_loss(outs, target)
		if self.loss_fn_name == 'BCE':
			target = target.float().unsqueeze(1)
		return self.loss_fn(outs, target)


class MLP(Model):
	def __init__(
					self, layers, init_method='xavier_unif', loss_name='CE',
					make_funct=False, pca_layers=None, dp_ratio=0.3):
		super(MLP, self).__init__(
									loss_name=loss_name,
									make_funct=make_funct, pca_layers=pca_layers
								)
		sequence = []
		for i in range(len(layers) - 1):
			sequence.append(nn.Dropout(dp_ratio))
			sequence.append(nn.Linear(layers[i], layers[i + 1]))
			sequence.append(nn.ReLU())
		self.model = nn.Sequential(*sequence[:-1])  # [:-1] to remove the last relu
		self.model.apply(weight_init(init_method))


class WideResnet(Model):
	def __init__(
					self, out_class_dict, depth, widen_factor, loss_name='CE',
					make_funct=False, pca_layers=None, dropRate=0.0
				):
		super(WideResnet, self).__init__(
									loss_name=loss_name,
									make_funct=make_funct, pca_layers=pca_layers
								)
		self.model = WideResNet(depth, out_class_dict, widen_factor=widen_factor, dropRate=dropRate)
		self.model.apply(weight_init('kaiming_normal'))


class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class TransferResnet(nn.Module):
	def __init__(self, out_class_dict, base_resnet='18', pretrained=True, dp_ratio=0.1):
		super(TransferResnet, self).__init__()
		if base_resnet == '18':
			body = models.resnet18(pretrained=pretrained)
			self.logit_sz = 512
		elif base_resnet == '50':
			self.logit_sz = 2048
			body = models.resnet50(pretrained=pretrained)
		else:
			raise ValueError('Wrong ResNet type specified')
		body.fc = Identity()
		for head_name, num_classes in out_class_dict.items():
			this_head = nn.Linear(self.logit_sz, num_classes)
			self.add_module("fc-{}".format(head_name), this_head)
		self.body = body
		self.dp_ratio = dp_ratio

	def body_forward(self, x):
		x = self.body.conv1(x)
		x = self.body.bn1(x)
		x = self.body.relu(x)
		x = self.body.maxpool(x)
		x = F.dropout(x, p=self.dp_ratio, training=self.training)

		x = self.body.layer1(x)
		x = F.dropout(x, p=self.dp_ratio, training=self.training)
		x = self.body.layer2(x)
		x = F.dropout(x, p=self.dp_ratio, training=self.training)
		x = self.body.layer3(x)
		x = F.dropout(x, p=self.dp_ratio, training=self.training)
		x = self.body.layer4(x)

		x = self.body.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.body.fc(x)
		return x

	def forward(self, x, head_name=None):
		logits = self.body_forward(x)
		if self.training:
			logits = F.dropout(logits, p=self.dp_ratio)
		this_fc = getattr(self, "fc-{}".format(head_name), None)
		assert this_fc is not None, 'Need to give a valid headname {}'.format(head_name)
		return this_fc(logits)


class ResNet(Model):
	def __init__(
					self, out_class_dict, base_resnet='18',
					make_funct=False, pca_layers=None, dropRate=0.0, pretrained=True
				):
		loss_name = 'BCEWLogits'
		super(ResNet, self).__init__(
									loss_name=loss_name,
									make_funct=make_funct, pca_layers=pca_layers
								)
		self.model = TransferResnet(out_class_dict, base_resnet=base_resnet, pretrained=pretrained, dp_ratio=dropRate)
		self.src_loss = self.get_loss_fn('CE')
		self.is_chexnet = True
		if not pretrained:
			print('OVERRIDING THE PRETRAINED WEIGHTS')
			self.model.apply(weight_init('kaiming_normal'))
		else:
			print('USING THE PRETRAINED MODEL')


if __name__ == '__main__':
	model = ResNet({'src': 1000, 'tgt': 5})
