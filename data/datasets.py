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
import numpy as np
import torchvision
from collections import defaultdict
import matplotlib.pyplot as plt
import math
import pdb
import os
from glob import glob
import gc


IMGNET_PATH = "/projects/tir4/users/ldery/tiny_imagenet/tiny-imagenet-200/train"
CHXPERT_PATH = "/projects/tir4/users/ldery/chexpert_dataset/CheXpert-v1.0-small"

def add_dataset_opts(parser):

	parser.add_argument('-train-perclass', type=int, default=100, help='Number of training examples per class')
	parser.add_argument('-val-perclass', type=int, default=15, help='Number of validation examples per class')
	parser.add_argument('-test-perclass', type=int, default=500, help='Number of test examples per class')
	parser.add_argument('-tgt-dgts', type=str, default='[6, 7, 14, 18, 24]') # Use when dealing with MNIST
	parser.add_argument('-dataset-path', type=str, default='.')
	parser.add_argument('-dataset-type', type=str, choices=['MNIST', 'CIFAR10', 'CIFAR100', 'CHEXPERT'], default='MNIST')
	parser.add_argument('-shots', type=int, default=1, help='num tgt per-class to include in source train')
	parser.add_argument('-num-monitor', type=int, default=10, help='number of examples per-class for monitoring val loss')
	parser.add_argument('-tiny-imagenet-loc', type=str, default=IMGNET_PATH)
	parser.add_argument('-chexpert-loc', type=str, default=CHXPERT_PATH)
	parser.add_argument('-imgnet-per-class', type=int, default=250)
	parser.add_argument('-imgnet-n-classes', type=int, default=200)
	return parser


# This converts numpy array data to a pytorch tensor
def to_tensor(data):
	if not len(data[0]):
		return None, None
	data = [torch.stack(data[0]), torch.tensor(data[1])]
	if torch.cuda.is_available():
		data[0] = data[0].cuda()
		data[1] = data[1].cuda()
	return data


# Super-class for all dataset manipulations
class Dataset(object):
	# Initialization for dataset object
	def __init__(self, tgt_dgts, **kwargs):
		self.shots = kwargs['shots']
		self.tgt_dgts = tgt_dgts
		self.shuffle = kwargs['shuffle'] if 'shuffle' in kwargs else False
		self.num_monitor = kwargs['num_monitor']  # the number of samples per-class to use for earlystopping monitoring
		self.val_taskweight = kwargs['val_taskweight']  # for upsampling target sample set during multitasking

	# Re-index labels based on super-class
	def update_ys(self, set_, group_):
		if hasattr(self, 'is_chexpert'):
			# This is a no-op when dealing with the chexpert dataset
			return set_
		if group_ == 'tgt':
			group = self.tgt_dgts
		else:
			group = self.src_dgts
		ys = set_[1]
		new_ys = torch.tensor([group.index(k) for k in set_[1]])
		new_ys = new_ys.to(ys.device)
		return [set_[0], new_ys]

	def _group_data(self, src_dgts, tgt_dgts, num_tr, num_val, num_test, flatten=True):
		# In the case where the target classes are a subset of the source classes
		self.src_dgts = list(set(src_dgts) - set(tgt_dgts))

		# This function groups the dataset according to the desired digits
		self.src_train = [[], []]
		self.src_valid = [[], []]

		self.tgt_valid = [[], []]
		self.tgt_test = [[], []]

		src_dgt_counts = np.array([0] * len(src_dgts))
		data_dict = defaultdict(list)
		for x, y in self.train:
			to_append = x.flatten() if flatten else x

			# We've filled the buffer for number of training examples for this class
			if np.all(src_dgt_counts == num_tr) and (y in self.src_dgts):
				continue

			if (y in self.src_dgts) and src_dgt_counts[src_dgts.index(y)] < num_tr:
				src_dgt_counts[src_dgts.index(y)] += 1
				self.src_train[0].append(to_append)
				self.src_train[1].append(y)

			if y in tgt_dgts:
				data_dict[y].append(to_append)

		# Group the test data accordning to the labels for easy sampling
		for x, y in self.test:
			to_append = x.flatten() if flatten else x
			if y in tgt_dgts:
				self.tgt_test[0].append(to_append)
				# add num_val of this class to the target test data
				self.tgt_test[1].append(y)
			else:
				data_dict[y].append(to_append)

		monitor_val = [[], []]
		for k, v in data_dict.items():
			num_samples = 0
			if k in self.src_dgts:
				assert k not in self.tgt_dgts, ' Invalid member of Target Class Set'
				num_val_src = len(v)
				num_samples += num_val_src
			if k in self.tgt_dgts:
				assert k not in self.src_dgts, ' Invalid member of Source Class Set'
				# we are gathering data for the target class
				num_samples += (num_val + self.num_monitor)
			chosen = np.random.choice(len(v), size=num_samples, replace=False)
			chosen = [v[i_] for i_ in chosen]
			start_ = 0
			if k in self.src_dgts:
				assert k not in self.tgt_dgts, ' Invalid member of Target Class Set'
				self.src_valid[0].extend(chosen[:num_val_src])  # add num_val of this class to the validation data
				self.src_valid[1].extend([k] * num_val_src)
				start_ = num_val_src
			if k in self.tgt_dgts:
				assert k not in self.src_dgts, ' Invalid member of Source Class Set'
				self.tgt_valid[0].extend(chosen[start_:(start_ + num_val)])  # add num_val of this class to the target val data
				self.tgt_valid[1].extend([k] * num_val)  # use index in the target dgt set
				monitor_val[0].extend(chosen[(start_ + num_val):])
				monitor_val[1].extend([k] * self.num_monitor)
		assert len(monitor_val[0]) == len(monitor_val[1]), 'Mismatch in monitor_val example and label sizes'
		self.monitor_val = to_tensor(monitor_val)
		self.ref_val = to_tensor(self.tgt_valid)  # This is the batch we are going to use for subspace alignment

	def _get_iterator(self, dataset, batch_sz, use_group_labels, is_multitask=False, shuffle=True):
		def get_batch(idxs, xs, ys, group=None):
			this_xs = torch.stack([xs[j] for j in idxs])
			if use_group_labels:
				# We want to use the labels in the target set instead of the original names in the source set
				assert not hasattr(self, 'is_chexpert'), 'For chexpert we have to use the tgt labels directly'
				if group is None:
					group = self.tgt_dgts if ys[0] in self.tgt_dgts else self.src_dgts
				assert ys[0] in group, 'This label isnt in this group'
				this_ys = torch.tensor([group.index(ys[j]) for j in idxs])
			else:
				if torch.is_tensor(ys[0]):
					this_ys = torch.stack([ys[j] for j in idxs])
				else:
					this_ys = torch.tensor([ys[j] for j in idxs])
			return this_xs, this_ys

		xs, ys = dataset if not is_multitask else dataset[0]
		if not len(xs):
			return
		idxs = list(range(len(xs))) if not shuffle else np.random.permutation(len(xs))
		# Check if we have enough data to make batches of this size
		num_batches = len(xs) * 1.0 / batch_sz
		if num_batches < 1.0:
			batch_sz = len(xs) // 2  # default to just 2 batches
			assert batch_sz > 0, 'Not enough examples to construct batches'
			num_batches = len(xs) * 1.0 / batch_sz
		num_batches = math.ceil(num_batches)
		if is_multitask:
			v_xs, v_ys = dataset[1]
			# Upsample to the size of the training set
			v_batch_sz = math.ceil(math.ceil(len(xs) * self.val_taskweight) / num_batches)
			size_ = v_batch_sz * num_batches
			v_idxs = np.random.choice(len(v_xs), size=size_, replace=True)

		for i in range(num_batches):
			tr_idxs = idxs[(i * batch_sz): (i + 1) * batch_sz]  # get the indices for this batch
			tr_xs, tr_ys = get_batch(tr_idxs, xs, ys)
			if not is_multitask:
				if torch.cuda.is_available():
					yield tr_xs.cuda(), tr_ys.cuda()
				else:
					yield tr_xs, tr_ys
			else:
				val_idxs = v_idxs[(i * v_batch_sz): (i + 1) * v_batch_sz]
				val_xs, val_ys = get_batch(val_idxs, v_xs, v_ys)
				if torch.cuda.is_available():
					yield (tr_xs.cuda(), val_xs.cuda()), (tr_ys.cuda(), val_ys.cuda())
				else:
					yield (tr_xs, val_xs), (tr_ys, val_ys)

	def get_dataset(self, types, tgt_labels_only):
		dataset = [[], []]
		for type_ in types.split("|"):
			if type_ == 'src_train':
				this_set = self.src_train
			elif type_ == 'src_valid':
				this_set = self.src_valid
			elif type_ == 'tgt_valid':
				this_set = self.tgt_valid
			elif type_ == 'tgt_test':
				this_set = self.tgt_test
			elif type_ == 'monitor_val':
				this_set = self.monitor_val
			if not tgt_labels_only:
				dataset[0].extend(this_set[0])
				dataset[1].extend(this_set[1])
			else:
				# Only extract from this dataset, examples that have labels in the tgt set
				assert not hasattr(self, 'is_chexpert'), 'For chexpert we have to use the tgt labels directly'
				for x, y in zip(*this_set):
					if y in self.tgt_dgts:
						dataset[0].append(x)
						dataset[1].append(y)
		return dataset

	# Get an iterator through the dataset
	def get_iter(self, batch_sz, type_='src_train', tgt_labels_only=False):
		# dataset_types seperated by "." means we are multitasking
		is_multi = '.' in type_
		if is_multi:  # We are doing multitask
			dataset = []
			for sub_type_ in type_.split('.'):
				set_ = self.get_dataset(sub_type_, tgt_labels_only)
				dataset.append(set_)
		else:
			dataset = self.get_dataset(type_, tgt_labels_only)
		use_grp_labels = True
		if hasattr(self, 'is_chexpert'):
			use_grp_labels = False
		return self._get_iterator(dataset, batch_sz, use_grp_labels, shuffle=self.shuffle, is_multitask=is_multi)


class MNIST(Dataset):
	def __init__(
					self, src_dgts, tgt_dgts,
					num_train_perclass, num_val_perclass,
					num_test_perclass, **kwargs
				):
		# Get the mnist dataset from torchvision
		super(MNIST, self).__init__(tgt_dgts, **kwargs)
		save_path = "~/" if 'save_path' not in kwargs else kwargs['save_path']
		tform = torchvision.transforms.Compose([
								torchvision.transforms.ToTensor(),
							])
		self.train = torchvision.datasets.MNIST(save_path, train=True, download=True, transform=tform)
		self.test = torchvision.datasets.MNIST(save_path, train=False, download=True, transform=tform)
		self._group_data(src_dgts, tgt_dgts, num_train_perclass, num_val_perclass, num_test_perclass)


class CIFAR10(Dataset):
	def __init__(
					self, src_dgts, tgt_dgts,
					num_train_perclass, num_val_perclass,
					num_test_perclass, **kwargs
				):
		# Get the cifar10 dataset from torchvision
		super(CIFAR10, self).__init__(tgt_dgts, **kwargs)
		save_path = "~/" if 'save_path' not in kwargs else kwargs['save_path']
		normalize = torchvision.transforms.Normalize(
											mean=[0.50707516, 0.48654887, 0.44091784],
											std=[0.26733429, 0.25643846, 0.27615047]
										)
		tform = torchvision.transforms.Compose([
								torchvision.transforms.ToTensor(),
								normalize
							])
		self.train = torchvision.datasets.CIFAR10(save_path, train=True, download=True, transform=tform)
		self.test = torchvision.datasets.CIFAR10(save_path, train=False, download=True, transform=tform)
		self._group_data(src_dgts, tgt_dgts, num_train_perclass, num_val_perclass, num_test_perclass, flatten=False)


class CIFAR100(Dataset):
	def __init__(
					self, src_dgts, tgt_dgts,
					num_train_perclass, num_val_perclass,
					num_test_perclass, **kwargs
				):
		# Get the cifar100 dataset from torchvision
		super(CIFAR100, self).__init__(tgt_dgts, **kwargs)
		save_path = "~/" if 'save_path' not in kwargs else kwargs['save_path']
		normalize = torchvision.transforms.Normalize(
											mean=[0.4914, 0.4822, 0.4465],
											std=[0.2023, 0.1994, 0.2010]
										)
		tform = torchvision.transforms.Compose([
								torchvision.transforms.ToTensor(),
								normalize
							])
		self.train = torchvision.datasets.CIFAR100(save_path, train=True, download=True, transform=tform)
		self.test = torchvision.datasets.CIFAR100(save_path, train=False, download=True, transform=tform)
		self._group_data(src_dgts, tgt_dgts, num_train_perclass, num_val_perclass, num_test_perclass, flatten=False)


def tf_init_transform(imgs_sz):
	import tensorflow as tf

	def load_img(file_path):
		img = tf.io.read_file(file_path)
		img = tf.image.decode_jpeg(img, channels=3)
		img = tf.image.convert_image_dtype(img, tf.float32)
		img = tf.image.resize(img, [imgs_sz[0], imgs_sz[1]])
		return img.numpy()

	return load_img


def tf_tensor_transform():
	normalize = torchvision.transforms.Normalize(
					mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
				)

	def transform(img):
		img = img.transpose((2, 0, 1))
		img = normalize(torch.tensor(img))
		return img
	return transform


CHXPERT_IMGSZ = (224, 224)
CHEXPERT_CLASSES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


class ChexPert_n_Imagenet(Dataset):
	def __init__(
					self, imgnet_per_class, n_src_classes,
					num_train, num_monitor,
					**kwargs
				):
		n_test_classes = 5
		self.src_dgts = range(n_src_classes)
		self.tgt_dgts = range(n_test_classes)
		kwargs['num_monitor'] = num_monitor
		super(ChexPert_n_Imagenet, self).__init__(self.tgt_dgts, **kwargs)
		imgnet_save_path = kwargs['imgnet_save_path']
		assert imgnet_save_path is not None, 'No Path to Imagenet Specified. Need to set IMGNET_PATH global variable in datasets.py'
		chexpert_save_path = kwargs['chexpert_save_path']
		assert chexpert_save_path is not None, 'No Path to Chexpert Specified. Need to set CHXPERT_PATH global variable in datasets.py'
		self.init_transform = tf_init_transform(CHXPERT_IMGSZ)
		self.tf_tensor_transform = tf_tensor_transform()
		self.get_chexpert(chexpert_save_path, num_train, num_monitor)
		self.get_imagenet(imgnet_save_path, n_src_classes, imgnet_per_class)
		self.is_chexpert = True  # So that we can decide not to use special labels

	def _get_imgs(self, fnames):
		imgs = []
		for path in fnames:
			img = self.init_transform(path)  # Original Maml code does inversion so inverting too
			imgs.append(self.tf_tensor_transform(img))
		return imgs

	def get_chexpert_imgs(self, csv_path, n_examples=-1):
		examples = np.loadtxt(csv_path, dtype=str, delimiter=',', skiprows=1)
		headers = open(csv_path, 'r').readline().split(',')
		classes = CHEXPERT_CLASSES
		class_idxs = [headers.index(class_) for class_ in classes]
		n_examples = n_examples if n_examples > 0 else len(examples)
		chosen_idxs = set()
		# Get the number of examples per-class
		per_class = int(n_examples / len(classes))
		for idx_ in class_idxs:
			this_class = examples[:, idx_]
			pos_idxs = set(np.nonzero(this_class == '1.0')[0])
			neg_idxs = set(np.nonzero(this_class == '0.0')[0])
			pos_list = list((pos_idxs - neg_idxs) - chosen_idxs)
			neg_list = list((neg_idxs - pos_idxs) - chosen_idxs)

			nsamples = min(int(per_class / 2), len(pos_list))
			pos_chosen = np.random.choice(pos_list, size=nsamples, replace=False)

			nsamples = min(int(per_class / 2), len(neg_list))
			neg_chosen = np.random.choice(neg_list, size=nsamples, replace=False)
			chosen_idxs.update(pos_chosen)
			chosen_idxs.update(neg_chosen)
		print('Actually got {}. {} samples requested'.format(len(chosen_idxs), n_examples))
		fnames, ys = [], []
		examples[examples == ''] = '0.0'
		examples[examples == '-1.0'] = '1.0'  # This choice is arbitray apparently from past chexpert papers.
		for idx in chosen_idxs:
			entry = examples[idx]
			new_fname = os.path.join(CHXPERT_PATH, '/'.join(entry[0].split('/')[1:]))
			fnames.append(new_fname)
			this_y = [float(entry[c_idx]) for c_idx in class_idxs]
			ys.append(this_y)
		imgs = self._get_imgs(fnames)
		return [imgs, ys]

	def get_chexpert(self, save_path, num_tr, num_monitor):
		# Get the data
		self.tgt_valid = self.get_chexpert_imgs(os.path.join(save_path, 'train.csv'), n_examples=num_tr)
		self.tgt_test = self.get_chexpert_imgs(os.path.join(save_path, 'valid.csv'), n_examples=-1)  # change back to -1
		monitor_val = self.get_chexpert_imgs(os.path.join(save_path, 'train.csv'), n_examples=num_monitor)

		self.monitor_val = to_tensor(monitor_val)
		self.ref_val = to_tensor(self.tgt_valid)  # This is the batch we are going to use for subspace alignment

		print('TGT VALID   Data :  {} egs'.format(len(self.tgt_valid[0])))
		print('TGT TEST    Data :  {} egs'.format(len(self.tgt_test[0])))
		print('TGT MONITOR Data :  {} egs'.format(len(self.monitor_val[0])))
		gc.collect()

	def get_imagenet(self, save_path, n_src_classes, imgnet_per_class):
		fldrs = glob(os.path.join(save_path, "*"))
		err_msg = "{} src classes specified but only {} are available".format(n_src_classes, len(fldrs))
		assert len(fldrs) >= n_src_classes, err_msg
		fldrs = fldrs[:n_src_classes]
		self.src_train = [[], []]
		for class_, fldr in enumerate(fldrs):
			if class_ % int((len(fldrs) / 10)) == 0:
				print('Currently working on class : {}/{}'.format(class_ + 1, len(fldrs)))
				gc.collect()

			files = glob(os.path.join(fldr, 'images/*'))
			if len(files) > imgnet_per_class:
				files = [files[x] for x in np.random.choice(len(files), size=imgnet_per_class, replace=False)]
			imgs = self._get_imgs(files)
			ys = [class_] * len(imgs)
			self.src_train[0].extend(imgs)
			self.src_train[1].extend(ys)
		self.src_valid = self.src_train
		print('Train Data :  {} egs'.format(len(self.src_train[0])))
