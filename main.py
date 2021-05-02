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

import numpy as np
import torch
from argparse import ArgumentParser
from models.models import add_model_opts, MLP, WideResnet, ResNet
from data.datasets import add_dataset_opts, MNIST, CIFAR10, CIFAR100, ChexPert_n_Imagenet, CHEXPERT_CLASSES
from data.superclasses import cifar100_super_classes
from algorithms.algos import Trainer, add_trainer_args
from algorithms.utils import set_model_dropout
from os import path
from copy import deepcopy
from collections import defaultdict
from vis import vis_loss_acc, vis_delta_norm
import time
import os
import random
import pickle as pkl
import gc

MY_CHEXPERT_CLASSES = ['ALL']
MY_CHEXPERT_CLASSES.extend(CHEXPERT_CLASSES)


# Setup the arguments
def get_options():
	parser = ArgumentParser(description='Code for Auxiliary Task Update Decomposition')
	parser.add_argument('-log-comment', type=str, default='experiment')
	parser.add_argument('-seed', type=int, default=1)
	parser.add_argument('-num-runs', type=int, default=4, help='number of reruns so we can estimate confidence interval')
	parser.add_argument('-no-src-only', action='store_true', help='do not train the source only model')
	parser.add_argument('-no-tgt-only', action='store_true', help='do not train the target only model')
	parser.add_argument(
							'-ft-model', default='none', type=str,
							choices=['src', 'src_pca', 'none'], help='model to finetune. src finetunes on model trained only on the source distribution. src_pca finetunes model that was trained on source distribution using ATTITTUD updates'
						)
	parser.add_argument('-exp-name', default='test_run', type=str, help='name of this experiment')
	parser.add_argument('-eta-exp', action='store_true', help='Vary the projection coefficients eta. Thus using -eta-exp  "(1, 1, 0)|(1, -1, 1)" will run num_runs experiments for each of these 2 configurations of eta  ')
	parser.add_argument('-use-jvp', action='store_true', help='use jvp for finding the subspace components')
	parser.add_argument('-multi-cifar', action='store_true', help='do multicifar100 experiments on superclasses. Assumes num-runs is the number of multicifar100 classes in consideration')
	parser.add_argument('-data-seed', type=int, default=1234, help='The seed to use for the dataset')
	add_model_opts(parser)
	add_dataset_opts(parser)
	add_trainer_args(parser)
	opts = parser.parse_args()
	return opts


# Runs a given model
def run_model(
				opts, model, dataset, save_fldr, proj_lambda_="(1.0, 1.0, 1.0)",
				pca_grad=False, trainset_id='src_train', valset_id='src_valid',
				use_tgt_labels=False, tgt_labels_only=False, viz=True, ft=False,
				set_out_classes=True, num_classes=10, is_tgt_only=False, use_last=False
			):
	# Setup the module to do training
	do_multitask = opts.do_multitask if not ft else False
	do_pcgrad = opts.do_pcgrad if not ft else False
	is_chexnet = (opts.dataset_type == 'CHEXPERT')
	trainer = Trainer(
						opts.train_epochs, opts.chkpt_path, opts.patience,
						proj_lambda_=proj_lambda_, pca_grad=pca_grad,
						pca_layers=eval(opts.pca_layers), num_pca_basis=opts.num_pca_basis,
						pca_nsamples=opts.pca_nsamples, pca_every=opts.pca_every,
						do_multitask=do_multitask, min_tr_loss=opts.min_tr_loss,
						val_taskweight=opts.val_taskweight, lowrank_nsamples=opts.lowrank_nsamples,
						do_pcgrad=do_pcgrad, is_chexnet=is_chexnet
					)
	# If we are doing pca, we need to get the reference target set as well
	#  as a monitor set so we can see how it's performance is affected.
	# Reference set => set of examples we use to create subspace for gradient decomposition
	# Monitor set => set of examples we use for validation gradient monitoring
	if pca_grad:
		ref_val = dataset.update_ys(dataset.ref_val, 'tgt')
		monitor_val = dataset.update_ys(dataset.monitor_val, 'tgt')
		trainer.setup_ref_n_monitor_vals(ref_val, monitor_val)

	optim, scheduler = trainer.get_optim(model, opts, ft=ft)
	if not path.exists(save_fldr):
		os.makedirs(save_fldr)
	save_path = path.join(save_fldr, "model.chkpt")
	# If we are training on the target distribution or finetuning then use a special batchsize
	# This is because the source data and target data might have different sizes. They are the same by default
	batch_sz = opts.tgt_batch_sz if ft or 'tgt' in trainset_id else opts.src_batch_sz
	metrics, best_val = trainer.train(
										model, dataset, optim, lr_scheduler=scheduler,
										batch_sz=batch_sz, trainset_id=trainset_id,
										valset_id=valset_id, use_tgt_labels=use_tgt_labels,
										tgt_labels_only=tgt_labels_only, model_chkpt_path=save_path,
										num_classes=num_classes, out_classes=eval(opts.tgt_dgts),
										normal_train_iters=opts.normal_train_iters, not_is_ft=not ft,
										is_tgt_only=is_tgt_only, use_last=use_last
								)
	pkl.dump(metrics, open(path.join(save_fldr, "train_metrics.pkl"), "wb"))
	# Visualize the results
	if viz:
		save_path = path.join(save_fldr, "metric_results.png")
		vis_loss_acc(metrics, save_path)
		if pca_grad:
			save_path = path.join(save_fldr, "delta_norms_pca.png")
			vis_delta_norm(metrics, save_path)
	tgt_test_iter = dataset.get_iter(
										opts.tgt_batch_sz, type_='tgt_test',
										tgt_labels_only=tgt_labels_only
									)
	_, model_to_tgt_test = trainer.run_epoch(model, tgt_test_iter, None, group='tgt')
	if is_chexnet:
		model_to_tgt_test = trainer.get_auc(reset=True)
	return model_to_tgt_test


def get_model(opts, num_classes, layers=None, make_funct=False, pca_layers=None):
	num_tgt_classes = len(eval(opts.tgt_dgts))
	out_class_dict = {'tgt': num_tgt_classes}
	if num_classes - num_tgt_classes > 0:
		out_class_dict['src'] = num_classes - num_tgt_classes
	if opts.model_type == 'MLP':
		model = MLP(layers, make_funct=make_funct, pca_layers=pca_layers)
	elif opts.model_type == 'WideResnet':
		model = WideResnet(
							out_class_dict, opts.depth, opts.widen_factor, dropRate=opts.dropRate,
							make_funct=make_funct, pca_layers=pca_layers
						)
	elif opts.model_type == 'ResNet':
		# Doing this specially for imagenet
		out_class_dict = {'src': opts.imgnet_n_classes, 'tgt': 5}
		model = ResNet(
						out_class_dict, base_resnet=opts.base_resnet,
						make_funct=make_funct, pca_layers=pca_layers, pretrained=opts.pretrained,
						dropRate=opts.dropRate
					)
	else:
		raise ValueError

	# Do Low-Rank approximation on all the layers of the network that have parameters
	# This code gathers all the layer names so we can apply ATTITTUD to all layers
	# Modify this if we want to apply ATTITTUD to a smaller subset of the layers
	pca_layers = []
	total_el = 0
	for k, v in model.named_parameters():
		if 'fc' in k:
			# Print out these statistics so we know that we have the same initialization across all runs
			stats = v.min().item(), v.mean().item(), v.max().item(), v.std().item()
			print("{} | min {}, mean {}, max {}, std {}".format(k, *stats))
		total_el += v.numel()
		layer_name = ".".join(k.split(".")[:-1])
		if layer_name not in pca_layers:
			pca_layers.append(layer_name)
	print('Model has {}M Parameters'.format(total_el / 1e6))
	opts.pca_layers = str(pca_layers)
	model.pca_layers = pca_layers
	return model


def get_dataset(opts):
	# Note num_classes depends on the dataset
	num_classes = 10
	if opts.dataset_type == 'MNIST':
		dataset = MNIST(
							range(num_classes), eval(opts.tgt_dgts),
							opts.train_perclass, opts.val_perclass,
							opts.test_perclass, shuffle=True, save_path=opts.dataset_path,
							shots=opts.shots, num_monitor=opts.num_monitor,
							val_taskweight=opts.val_taskweight
					)
	elif opts.dataset_type == 'CIFAR10':
		dataset = CIFAR10(
							range(num_classes), eval(opts.tgt_dgts),
							opts.train_perclass, opts.val_perclass,
							opts.test_perclass, shuffle=True, save_path=opts.dataset_path,
							shots=opts.shots, num_monitor=opts.num_monitor,
							val_taskweight=opts.val_taskweight
					)
	elif opts.dataset_type == 'CIFAR100':
		num_classes = 100
		dataset = CIFAR100(
							range(num_classes), eval(opts.tgt_dgts),
							opts.train_perclass, opts.val_perclass,
							opts.test_perclass, shuffle=True, save_path=opts.dataset_path,
							shots=opts.shots, num_monitor=opts.num_monitor,
							val_taskweight=opts.val_taskweight
					)
	elif opts.dataset_type == 'CHEXPERT':
		num_tgt_classes = 5
		num_classes = 1000 # Number of classes in pre-trained imagenet
		opts.tgt_dgts = str(range(num_tgt_classes))
		dataset = ChexPert_n_Imagenet(
					opts.imgnet_per_class, opts.imgnet_n_classes,
					opts.train_perclass * num_tgt_classes, opts.num_monitor,
					imgnet_save_path=opts.tiny_imagenet_loc,
					chexpert_save_path=opts.chexpert_loc, shuffle=True,
					shots=opts.shots, val_taskweight=opts.val_taskweight
				)
	else:
		raise ValueError
	return dataset, num_classes


def set_random_seed(seed):
	# Esp important for ensuring deterministic behavior with CNNs
	torch.backends.cudnn.deterministic = True
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	cuda_available = torch.cuda.is_available()
	if cuda_available:
		torch.cuda.manual_seed_all(seed)
	return cuda_available


def print_results(run_dict):
	print('Printing results So Far')
	for k, v in run_dict.items():
		v = np.array(v)
		avg = v.mean(axis=0, keepdims=True)
		ci95 = (1.96 * np.std(v, axis=0, keepdims=True)) / np.sqrt(len(v))
		if avg.shape[-1] == 1:
			print("{:.3f} +/- {:.4f} = [{} | Tgt]".format(avg[0, 0], ci95[0, 0], k))
			print("{:.3f} +/- {:.4f} = [{} | Src]".format(avg[0, 1], ci95[0, 1], k))
		else:
			# We are dealing with CHEXPERT dataset and want to print out performance for each pathology too
			str_ = ["{:20} {:.3f} +/- {:.4f}".format(z, x, y) for x, y, z in zip(avg[0], ci95[0], MY_CHEXPERT_CLASSES)]
			str_ = "\n".join(str_)
			print("[{} | Tgt]\n{}".format(k, str_))
	print('\n')

def main():
	opts = get_options()
	print(opts)
	run_dict = defaultdict(list)
	opts.chkpt_path = path.join(opts.chkpt_path, opts.exp_name.replace(" ", "_"))
	num_iters = opts.num_runs

	# In general, we use | to seperate a list of possible values for a hyper-parameter
	if opts.multi_cifar:
		superclass_keys = list(cifar100_super_classes.keys())
		num_iters = len(superclass_keys)
	elif opts.eta_exp:
		projs = opts.proj_lambda_.split("|")
		projs = projs * opts.num_runs  # We want to run num_run experiments for each value of eta.
		num_iters = len(projs)
		print('List of eta projection coefficients to use : ', projs)
	elif opts.do_multitask:
		# 
		task_weights = opts.val_taskweight.split("|")
		task_min_tr = opts.min_tr_loss.split("|")
		all_weight_min_pairs = [(float(w), float(m)) for w in task_weights for m in task_min_tr] * opts.num_runs
		print('These are all pairings', all_weight_min_pairs)
		num_iters = len(all_weight_min_pairs)

	train_epochs, finetune_epochs = opts.train_epochs, opts.finetune_epochs
	pcgrad_ext = "pc_grad" if opts.do_pcgrad else ""
	# Setting the data seed
	opts.seed = opts.data_seed
	# Fix the dataset seed
	cuda_available = set_random_seed(opts.seed)
	dataset, num_classes = get_dataset(opts)

	for seed in range(num_iters):
		print('Currently Working on seed : ', seed)

		# Fixed data
		if opts.multi_cifar:
			# Assumes for multi-cifar, we have a run for each super-class key we are interested in. 
			# NB : Use "-num-runs 20 -multi-cifar" if you want to run the full multi-cifar experiments
			superclass_key = superclass_keys[seed]
			opts.tgt_dgts = str(cifar100_super_classes[superclass_key])
			print('Working on : ', superclass_key)
			# Set the model initialization Seed
			seed = opts.data_seed
			opts.val_taskweight = float(opts.val_taskweight)
			opts.min_tr_loss = float(opts.min_tr_loss)
		elif opts.eta_exp:
			opts.proj_lambda_ = projs[seed]
			seed = seed // (len(projs) // opts.num_runs)  # Same seed for different proj_lambda_ so we have a fair comparison
		elif opts.do_multitask:
			opts.val_taskweight = all_weight_min_pairs[seed][0]
			opts.min_tr_loss = all_weight_min_pairs[seed][1]
			# Ensure all pairs share same set of random seeds
			seed = seed // (len(all_weight_min_pairs) // opts.num_runs)
		if not opts.do_multitask:
			opts.val_taskweight = float(opts.val_taskweight)
			opts.min_tr_loss = float(opts.min_tr_loss)

		dataset.val_taskweight = opts.val_taskweight
		opts.seed = seed
		print('Setting Random Init seed to : ', seed)
		# Fixing the initialization
		set_random_seed(opts.seed)
		#########################################################################
		#########################################################################
		if not opts.no_tgt_only:
			print("Train solely on the in-domain data")
			opts.train_epochs = train_epochs
			tgt_layers = eval(opts.layer_sizes)
			tgt_layers[-1] = len(eval(opts.tgt_dgts))
			tgt_model = get_model(opts, len(eval(opts.tgt_dgts)), layers=tgt_layers)
			if cuda_available:
				tgt_model.cuda()
			save_fldr = path.join(opts.chkpt_path, "{}/tgt_dist".format(superclass_key if opts.multi_cifar else opts.seed))
			train_out = run_model(
									opts, tgt_model, dataset, save_fldr, trainset_id='tgt_valid',
									valset_id='tgt_test', use_tgt_labels=True, tgt_labels_only=False,
									set_out_classes=False, num_classes=tgt_layers[-1], is_tgt_only=True
								)
			print('TGT-TGT : {}'.format(train_out))
			run_dict["Train-Target|Evaluate-Target"].append(train_out)
		#########################################################################
		#########################################################################
		ft_model = None
		if not opts.no_src_only:
			print("Train the model on the source distribution")
			opts.train_epochs = train_epochs
			src_model = get_model(opts, num_classes, layers=eval(opts.layer_sizes))
			if cuda_available:
				src_model.cuda()
			desc = "src_dist_vwgt_{}_mintr_{}_{}".format(opts.val_taskweight, opts.min_tr_loss, pcgrad_ext)
			save_fldr = path.join(opts.chkpt_path, "{}/{}".format(superclass_key if opts.multi_cifar else opts.seed, desc))
			train_out = run_model(
									opts, src_model, dataset, save_fldr, trainset_id='src_train',
									valset_id='tgt_valid', use_tgt_labels=False, tgt_labels_only=False,
									num_classes=num_classes, use_last=opts.use_last_chkpt
								)
			if len(eval(opts.tgt_dgts)) == 2:
				# Binary Classification : use if model couldn't figure out which class is which
				train_out = train_out if train_out > 0.5 else 1.0 - train_out
			print('SRC-TGT ({}) : {}'.format(desc, train_out))
			run_dict['Train-Source({})|Evaluate-Target'.format(desc)].append(train_out)
			ft_model = src_model if opts.ft_model == 'src' else ft_model
		#########################################################################
		#########################################################################
		if opts.pca_grad:
			print("Train the model on the source distribution with ATTITUD enabled")
			opts.train_epochs = train_epochs
			src_model_pca = get_model(
										opts, num_classes, layers=eval(opts.layer_sizes),
										make_funct=opts.use_jvp, pca_layers=eval(opts.pca_layers)
									)
			if cuda_available:
				src_model_pca.cuda()
			desc = "{}/pca_{}_nc_{}_vwgt_{}_mintr_{}".format(
																superclass_key if opts.multi_cifar else opts.seed, opts.proj_lambda_,
																opts.num_pca_basis, opts.val_taskweight, opts.min_tr_loss
															)
			save_fldr = path.join(opts.chkpt_path, desc)
			train_out = run_model(
									opts, src_model_pca, dataset, save_fldr, proj_lambda_=opts.proj_lambda_,
									pca_grad=opts.pca_grad, trainset_id='src_train', valset_id='tgt_valid',
									use_tgt_labels=False, tgt_labels_only=False, viz=True,
									num_classes=num_classes
								)
			if len(eval(opts.tgt_dgts)) == 2:
				# Binary Classification : use if model couldn't figure out which class is which
				train_out = train_out if train_out > 0.5 else 1.0 - train_out
			run_dict['Train-Source(PCA_{})|Evaluate-Target'.format(opts.proj_lambda_)].append(train_out)
			ft_model = src_model_pca if opts.ft_model == 'src_pca' else ft_model
			print('PCA-{} : {}'.format(opts.proj_lambda_, train_out))

		#########################################################################
		#########################################################################
		opts.train_epochs = finetune_epochs
		if ft_model is not None:
			print("Finetune prev_source model on tgt_valid")
			tgt_ft_model = deepcopy(ft_model)
			set_model_dropout(tgt_ft_model.model, opts.ft_dropRate)
			ext = "pca_{}_nc_{}_vwgt_{}_mintr_{}".format(
															opts.proj_lambda_, opts.num_pca_basis,
															opts.val_taskweight, opts.min_tr_loss
														)
			ext_no_pca = "no_pca_vwgt_{}_mintr_{}_{}".format(opts.val_taskweight, opts.min_tr_loss, pcgrad_ext)
			ext = ext if opts.ft_model == 'src_pca' else ext_no_pca
			save_fldr = path.join(
									opts.chkpt_path,
									"{}/finetune_src_{}_tgt".format(superclass_key if opts.multi_cifar else opts.seed, ext)
								)
			train_out = run_model(
									opts, tgt_ft_model, dataset, save_fldr, trainset_id='tgt_valid',
									valset_id='tgt_test', use_tgt_labels=False, tgt_labels_only=False,
									ft=True, num_classes=num_classes
								)
			print('Train-Source|Finetune-Target|Evaluate-Target : {}'.format(train_out))
			run_dict["Train-Source-{}|Finetune-Target|Evaluate-Target".format(ext)].append(train_out)
			#########################################################################
			#########################################################################
		print_results(run_dict)
	print('--' * 40)
	print_results(run_dict)
	# write the options and final performance to a file
	with open(path.join(opts.chkpt_path, "args_and_results.pkl"), 'wb') as handle:
		pkl.dump((opts, run_dict), handle)


if __name__ == '__main__':
	print('==' * 40 + "\n")
	print('BEGINNING RUN')
	print('==' * 40 + "\n")
	start = time.time()
	main()
	print('Main took a total of : ', time.time() - start)
