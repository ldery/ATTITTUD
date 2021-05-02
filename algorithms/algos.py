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
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from collections import defaultdict, namedtuple
from os import path
import pickle
import pdb
from sklearn import metrics
from .utils import *


DELTA = 1e-10
GradWeights = namedtuple('GradWeights', 'eta_tilde eta_pos eta_neg')


def add_trainer_args(parser):
	parser.add_argument('-train-epochs', type=int, default=150)
	parser.add_argument('-finetune-epochs', type=int, default=500)
	parser.add_argument('-patience', type=int, default=15)
	parser.add_argument('-lr-patience', type=int, default=4)
	parser.add_argument('-optimizer', type=str, default='Adam')
	parser.add_argument('-lr', type=float, default=1e-4)
	parser.add_argument(
							'-proj_lambda_', type=str, default="(1.0, 0.0, 0.0)",
							help="Weights of different components (tilde{eta}, eta_{+}, eta_{1}) for ATTITUD"
						)
	parser.add_argument('-pca-grad', action='store_true', help='If true, performs ATTITTUD for correcting gradients')
	parser.add_argument('-pca-layers', type=str, default="['model.0', 'model.2']", help='The list of layers to apply ATTITTUD to')
	parser.add_argument('-finetune-lr', type=float, default=5e-5)
	parser.add_argument('-src-batch-sz', type=int, default=16)
	parser.add_argument('-tgt-batch-sz', type=int, default=16)
	parser.add_argument('-chkpt-path', type=str, default='experiments', help='Where to store results of this experiment')
	parser.add_argument('-num-pca-basis', type=int, default=20, help='Size of subspace for operating on gradients via ATTITTUD')
	parser.add_argument('-pca-nsamples', type=int, default=30, help='Number of samples to use to estimate subspace')
	parser.add_argument('-lowrank-nsamples', type=int, default=16, help='Number of samples to estimate target task projection unto the subspace')
	parser.add_argument('-do-multitask', action='store_true', help='Treat as a multi-task loss')
	parser.add_argument(
							'-do-pcgrad', action='store_true',
							help='performs gradient deconflicting like in this paper : https://arxiv.org/pdf/2001.06782.pdf'
						)
	parser.add_argument('-val-taskweight', type=str, default='0.0', help='Task weighting when using multitasking. Note that we can use | to separate a list of weightings we wish to test')
	parser.add_argument('-min-tr-loss', type=str, default='0.0')
	parser.add_argument(
							'-normal-train-iters', type=int, default=0,
							help='How many epochs of normal training before starting pca-training'
						)
	parser.add_argument('-pca-every', type=int, default=1, help='How frequently to do pca')
	parser.add_argument('-use-last-chkpt', action='store_true', help='Instead of using best, use last checkpoint')
	parser.add_argument('-is-chexnet', action='store_true', help='We are using a chexnet model')
	parser.add_argument('-continue-from-last', action='store_true')


class Trainer(object):
	def __init__(
					self, train_epochs, chkpt_path, patience,
					proj_lambda_=(1.0, 1.0, 1.0), pca_grad=False, pca_layers=[],
					num_pca_basis=20, pca_nsamples=30, pca_every=1,
					do_multitask=False, min_tr_loss=0.0, val_taskweight=0.0,
					lowrank_nsamples=16, do_pcgrad=False, is_chexnet=False
				):
		self.chkpt_every = 20  # save to checkpoint
		self.train_epochs = train_epochs
		self.chkpt_path = chkpt_path
		self.patience = patience
		self.pca_every = pca_every
		self.pca_cntr = 0
		self.pca_grad = pca_grad
		self.do_multitask = do_multitask
		self.val_taskweight = val_taskweight
		self.max_grad_norm = 1.0
		self.min_tr_loss = min_tr_loss
		proj_lambda_ = eval(proj_lambda_)
		self.g_weights = GradWeights(
										eta_tilde=proj_lambda_[0], eta_pos=proj_lambda_[1],
										eta_neg=proj_lambda_[2]
									)
		self.pca_layers = pca_layers
		self.num_pca_basis = num_pca_basis
		self.pca_nsamples = pca_nsamples
		self.lowrank_nsamples = lowrank_nsamples
		self.do_pcgrad = do_pcgrad
		self.is_chexnet = is_chexnet
		self.roc_calc = [[], []]

	# Setup target set batch for doing subspace decomposition.
	# Monitor batch for validation set monitoring of the subspace created
	def setup_ref_n_monitor_vals(self, ref_val, monitor_val):
		self.reference_val = ref_val
		self.monitor_val = monitor_val

	def get_optim(self, model, opts, ft=False):
		lr = opts.lr if not ft else opts.finetune_lr
		if opts.optimizer == 'Adam':
			optim = Adam(model.parameters(), lr=lr)
		elif opts.optimizer == 'AdamW':
			print('Using AdamW Optimizer')
			optim = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
		else:
			raise ValueError
		# Reduce lr by 0.5 on plateau
		lr_scheduler = ReduceLROnPlateau(optim, factor=0.5, patience=opts.lr_patience, min_lr=1e-5)
		return optim, lr_scheduler

	# Calculate the number of correct predictions
	def _get_numcorrect(self, output, targets, mask=None, chxnet_src=False):
		with torch.no_grad():
			argmax = output.argmax(dim=-1).squeeze()
			if self.is_chexnet and (not chxnet_src):
				model_out = torch.nn.Sigmoid()(output)
				self.roc_calc[0].extend(targets.cpu().numpy())
				self.roc_calc[1].extend(model_out.cpu().numpy())
				targets = targets.view(-1)
				return (model_out.view(-1) > 0.5).eq(targets).sum() * 1.0 / output.shape[-1]
			if output.shape[-1] == 1:  # We have only 1 output - so sigmoid.
				argmax = (output > 0.5).squeeze().float()
			return argmax.eq(targets).sum()

	# Check if this layer is in the list of layers to apply ATTITUD to
	def _is_pca_layer(self, k):
		k = ".".join(k.split('.')[:-1])
		return k in self.pca_layers

	# Project the model gradients obtained from (base_set) unto the subspace spanned by ortho_basis 
	def get_low_rank(self, model, ortho_basis, base_set_x, base_set_y, head_name=None):
		loss = model.criterion(model(base_set_x, head_name=head_name), base_set_y)
		grads = list(torch.autograd.grad(loss, model.params, allow_unused=True))
		for idx, (g_, p_) in enumerate(zip(grads, model.params)):
			if g_ is None:
				grads[idx] = torch.zeros_like(p_)
		grad_vec = vectorize(grads)[1].detach()
		with torch.no_grad():
			low_rank = grad_vec.matmul(ortho_basis.t())
		grads = reshape_grad_vector(grad_vec, model.param_shapes())
		return low_rank, grads

	
# LDERY = todo => this is where you left off in cleaning up the code !!

	# get val gradients and do a low rank Orthonormal approx.
	def get_ortho_grad_basis(self, model, base_set, val_set):
		base_set_x, base_set_y = base_set
		proj_mat = torch.normal(mean=0.0, std=1.0, size=(self.pca_nsamples, self.num_pca_basis), device=base_set_x.device)
		idxs = np.random.choice(len(base_set_x), size=self.pca_nsamples, replace=False)
		subspace_set = base_set_x[idxs], base_set_y[idxs]
		# We may use a subset of the reference set
		prod_ = []
		# fix the dropout layers to 0
		old_dp = set_model_dropout(model.model, 0)
		for i in range(self.num_pca_basis):
			v_ = proj_mat[:, i]
			model.make_functional()
			out = torch.autograd.functional.vjp(model.functional_(subspace_set, head_name='tgt'), model.params, v=v_)
			prod_.append(vectorize(out[1])[1].unsqueeze(1))
		# return the dropout layers
		set_model_dropout(model.model, old_dp)
		with torch.no_grad():
			prod_ = torch.cat(prod_, dim=1)
			# My implementation of qr turned out to be faster
			# Seems the torch version makes underlying api calls which take time.
			ortho_basis = my_qr(prod_.t())

		# empirically found that re-sampling a set of examples for average gradient is better
		low_rank, grad_vec = self.get_low_rank(model, ortho_basis, val_set[0], val_set[1], head_name='tgt')
		return ortho_basis, (0.0, 1.0), low_rank, grad_vec

	def get_grads_and_proj(self, model, set_, V, stats, head_name='src', base_comp=None):
		input_x, y = set_
		x_out = model(input_x, head_name=head_name)
		chxnet_src = head_name == 'src'
		num_correct = self._get_numcorrect(x_out, y, chxnet_src=chxnet_src)
		loss = model.criterion(x_out, y, head_name=head_name)
		all_grads = list(torch.autograd.grad(loss, model.parameters(), allow_unused=True))
		this_grads, old_norms = [], []
		for idx_, (k, v) in enumerate(model.named_parameters()):
			if self._is_pca_layer(k):
				if all_grads[idx_] is None:
					all_grads[idx_] = torch.zeros_like(v)
				this_grads.append(all_grads[idx_])
				with torch.no_grad():
					old_norms.append(all_grads[idx_].norm())
		old_norms = np.array(old_norms)
		grad_shapes, grad_vec = vectorize(this_grads)
		with torch.no_grad():
			grad_vec = (grad_vec - stats[0]) / stats[1]
			grad_vec = grad_vec.unsqueeze(0)
			# project unto ortho basis
			low_rank = grad_vec.matmul(V.t())  # Train
			# low rank components where validation and train agree in direction
			mask_pos = ((low_rank * base_comp) > 0.0).float()
			new_grad_pos, pos_norm = get_new_grads(low_rank * mask_pos, V, grad_shapes, stats)
			new_grad_neg, neg_norm = get_new_grads(low_rank * (1.0 - mask_pos), V, grad_shapes, stats)

		with torch.no_grad():
			out_span = grad_vec.squeeze() - (new_grad_pos + new_grad_neg)
			final_grad = (out_span * self.g_weights.eta_tilde) + (new_grad_pos * self.g_weights.eta_pos) + \
				(new_grad_neg * self.g_weights.eta_neg)
			this_grads = reshape_grad_vector(final_grad, grad_shapes)
			new_norm = final_grad.norm()

		pca_idx_ = 0
		for idx_, (k, _) in enumerate(model.named_parameters()):
			if self._is_pca_layer(k):
				all_grads[idx_] = this_grads[pca_idx_]
				pca_idx_ += 1
		with torch.no_grad():
			old_norm = ((old_norms**2).sum()).sqrt()
		return loss, num_correct, (old_norm, new_norm, pos_norm, neg_norm), all_grads

	def get_proj_gradients(self, train_set, base_set, monitor_val, model, val_set=None):
		if val_set is None:
			idxs = np.random.choice(len(base_set[0]), size=self.lowrank_nsamples, replace=False)
			val_set = (base_set[0][idxs], base_set[1][idxs])

		self.pca_cntr += 1
		if (self.pca_every == self.pca_cntr) or (not hasattr(self, 'ortho_basis')):
			self.pca_cntr = 0
			ortho_basis, stats, self.base_comp, avg_val_grad = self.get_ortho_grad_basis(model, base_set, val_set)
			self.ortho_basis, self.stats = ortho_basis, stats
		else:
			ortho_basis, stats = self.ortho_basis, self.stats
			self.base_comp, avg_val_grad = self.get_low_rank(
															model, ortho_basis, val_set[0],
															val_set[1], head_name='tgt'
														)

		# get the gradients of each samples set and project to base
		train_grad_info = self.get_grads_and_proj(
							model, train_set, ortho_basis, stats,
							head_name='src', base_comp=self.base_comp
						)
		tr_loss, tr_num_crt, tr_delta_norm, tr_new_grads = train_grad_info

		m_val_grad_info = self.get_grads_and_proj(
							model, monitor_val, ortho_basis, stats,
							head_name='tgt', base_comp=self.base_comp
						)
		m_val_loss, val_num_crt, m_val_delta_norm, _, = m_val_grad_info

		# return modified gradients and statistics
		return tr_new_grads, (tr_delta_norm, tr_loss, tr_num_crt), (m_val_delta_norm, m_val_loss, val_num_crt), avg_val_grad

	def run_epoch_grad_proj(self, model, data_iter, optim, group=None):
		model.train()
		assert optim is not None, 'optimizer should be initialized'
		total_loss, total_correct, num_pts = 0.0, 0.0, 0.0
		# Setup to cache the norm statistics
		tr_norms, m_val_norms, m_val_loss = [], [], []

		for iter_, (x, y) in enumerate(data_iter):
			tr_set, val_set = (x, y), None
			if self.do_multitask:
				tr_set = x[0], y[0]
				val_set = x[1], y[1]
			results = self.get_proj_gradients(tr_set, self.reference_val, self.monitor_val, model, val_set=val_set)
			grads, tr_stats, m_val_stats, val_grads = results
			tr_norms.append(tr_stats[0])
			m_val_norms.append(m_val_stats[0])
			m_val_loss.append(m_val_stats[1])
			with torch.no_grad():
				losses = (tr_stats[1], 0.0)
				corrects = (tr_stats[2], 0.0)
				szs = (y[0].shape[0], y[1].shape[0]) if self.do_multitask else (y.shape[0], 0)
				out_ = self.get_multitask_weights(losses, corrects, szs)
				tr_weight, val_weight, loss, num_correct, sz = out_

				for tuple_ in zip(grads, val_grads, model.parameters(), model.names):
					grad, v_grad, param, name = tuple_
					if param.grad is None:
						param.grad = torch.zeros_like(grad)

					if 'fc' not in name:
						param.grad.data.copy_((grad * tr_weight) + (v_grad * val_weight))
					elif ('tgt' not in name):
						param.grad.data.copy_(grad * tr_weight)
					else:
						assert ('tgt' in name) and ('fc' in name), 'Wrong parameter is being updated'
						tgt_grad = v_grad * val_weight if self.do_multitask else v_grad
						param.grad.data.add_(tgt_grad)

				torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
			# Now do the normal optimizer step
			optim.step()
			total_loss += (loss * sz) / (tr_weight + DELTA)  # because the loss is an average
			total_correct += num_correct
			num_pts += sz

		tr_norms = np.array(tr_norms)
		tr_norms = (tr_norms / tr_norms[:, :1]).mean(axis=0)
		tr_norms = [n_.item() for n_ in tr_norms]
		m_val_norms = np.array(m_val_norms)
		m_val_norms = (m_val_norms / m_val_norms[:, :1]).mean(axis=0)
		m_val_norms = [n_.item() for n_ in m_val_norms]
		self.metrics['tr_grad_norm_frac'].append(tr_norms)
		self.metrics['m_val_grad_norm_frac'].append(m_val_norms)
		self.metrics['m_val_loss'].append(((sum(m_val_loss)) / len(m_val_loss)).item())
		if num_pts == 0:
			return -float('inf'), 0.0
		return (total_loss / num_pts).item(), (total_correct / num_pts).item()

	def run_model(self, x, y, model, group=None, reduct_='none'):
		model_out = model(x, head_name=group)
		loss_fn = model.get_loss_fn(model.loss_fn_name, reduction=reduct_)
		loss = loss_fn(model_out, y)
		num_correct = self._get_numcorrect(model_out, y)
		return loss, num_correct

	def get_vectorized_gradients(self, data, model, group='src'):
		loss, num_correct = self.run_model(data[0], data[1], model, group=group, reduct_='mean')
		all_grads = list(torch.autograd.grad(loss, model.parameters(), allow_unused=True))
		for idx_, p in enumerate(model.parameters()):
			if all_grads[idx_] is None:
				all_grads[idx_] = torch.zeros_like(p)
		_, grads = vectorize(all_grads)
		return loss, num_correct, grads

	def get_multitask_weights(self, losses, corrects, szs):
		tr_loss, val_loss = losses
		tr_crt, val_crt = corrects
		tr_sz, val_sz = szs

		tr_scale = 1.0 if tr_loss > self.min_tr_loss else 0.0
		sz = (tr_sz * tr_scale) + val_sz
		tr_weight = tr_sz * tr_scale / sz
		val_weight = val_sz / sz
		combined_loss = (tr_loss * tr_weight) + (val_loss * val_weight)
		combined_correct = (tr_crt * tr_scale) + val_crt
		return tr_weight, val_weight, combined_loss, combined_correct, sz

	def run_epoch(self, model, data_iter, optim, group=None):
		model.train()
		if optim is None:
			model.eval()
		total_loss, total_correct, num_pts = 0.0, 0.0, 0.0
		for x, y in data_iter:
			if self.do_multitask and (optim is not None):
				tr_loss, tr_crt, tr_grads = self.get_vectorized_gradients((x[0], y[0]), model, group='src')
				val_loss, val_crt, val_grads = self.get_vectorized_gradients((x[1], y[1]), model, group='tgt')

				out_ = self.get_multitask_weights((tr_loss, val_loss), (tr_crt, val_crt), (y[0].shape[0], y[1].shape[0]))
				tr_weight, val_weight, loss, num_correct, sz = out_

				with torch.no_grad():
					inner_prod = 0.0
					if self.do_pcgrad:
						inner_prod = tr_grads.dot(val_grads) / (torch.norm(val_grads)**2)
					tr_grads = (tr_grads - min(inner_prod, 0.0) * val_grads)
					tr_grads, val_grads = tr_grads * tr_weight, val_grads * val_weight
					final_grads = reshape_grad_vector((tr_grads + val_grads), model.param_shapes())
			else:
				loss, num_correct = self.run_model(x, y, model, group=group, reduct_='mean')
				sz = x.shape[0]
			if optim is not None:
				if self.do_multitask:
					for idx_, param in enumerate(model.parameters()):
						if param.grad is None:
							param.grad = torch.zeros_like(param)
						param.grad.data.copy_(final_grads[idx_])
				else:
					optim.zero_grad()
					loss.backward()
				torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
				optim.step()
			loss = loss.item()
			total_loss += (loss * sz)  # because the loss is an average
			total_correct += num_correct.item()
			num_pts += sz
		if num_pts == 0:
			return -float('inf'), 0.0
		return (total_loss / num_pts), (total_correct / num_pts)

	def model_exists(self, model, dataset, kwargs):
		chkpt_path = kwargs["model_chkpt_path"]
		dir_name = path.dirname(chkpt_path)
		metric_path = path.join(dir_name, "train_metrics.pkl")
		if not path.exists(metric_path):
			return None, None
		train_metrics = pickle.load(open(metric_path, 'rb'))
		best_val_loss = min(train_metrics['monitor_loss'][-(self.patience + 1):])
		final_path = self.get_chkpt_path('final', chkpt_path)
		if path.exists(final_path):
			model.load_state_dict(torch.load(final_path))
			return train_metrics, best_val_loss
		len_metrics = len(train_metrics['monitor_loss'])
		best_epoch = np.argmin(train_metrics['monitor_loss'][-(self.patience + 1):]) + len_metrics - (self.patience + 1)
		epochs_to_avg = list(range(best_epoch + 1, len_metrics))
		epochs_to_avg.append('best')
		model = self.chkpt_averaging(chkpt_path, model, epochs_to_avg, dataset, kwargs)
		# Save the model to the final path
		torch.save(model.state_dict(), final_path)
		return train_metrics, best_val_loss

	def get_chkpt_path(self, epoch, chkpt_path):
		this_path = chkpt_path.split('.')[:-1]
		this_path = ".".join(this_path)
		return this_path + "_epoch_" + str(epoch) + ".chkpt"

	def chkpt_averaging(self, chkpt_path, model, epochs_to_avg, dataset, kwargs):
		# need to get the train samples for fixing the batch normalization layers
		data_type = kwargs['trainset_id']
		if self.do_multitask and (kwargs['not_is_ft']):
			data_type = "{}.{}".format(kwargs['trainset_id'], kwargs['valset_id'])
		grp = 'tgt' if not kwargs['not_is_ft'] else 'src'
		train_iter = dataset.get_iter(
									kwargs['batch_sz'], type_=data_type,
									tgt_labels_only=kwargs['tgt_labels_only']
								)
		cur_model_sd = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
		num_chkpts_used = 0.0
		for epoch in epochs_to_avg:
			this_path = self.get_chkpt_path(epoch, chkpt_path)
			if not path.exists(this_path):
				continue
			num_chkpts_used += 1
			this_sd = torch.load(this_path)
			for id_, p_ in this_sd.items():
				cur_model_sd[id_].add_(p_)
		# load the state dict
		if num_chkpts_used > 0:
			for _, v in cur_model_sd.items():
				div_tensor = torch.tensor(num_chkpts_used, dtype=v.dtype)
				v.div_(div_tensor)
		model.load_state_dict(cur_model_sd)
		bn_update(train_iter, model, multitask=self.do_multitask, group=grp)
		return model

	def get_auc(self, reset=True):
		auc_per_class = metrics.roc_auc_score(self.roc_calc[0], self.roc_calc[1], average=None)
		average_auc = metrics.roc_auc_score(self.roc_calc[0], self.roc_calc[1])
		if reset:
			self.roc_calc = [[], []]
		return [average_auc, *auc_per_class]

	def train(self, model, dataset, optim, lr_scheduler=None, **kwargs):
		# Do the training
		# Check if the model already exists and just return it
		train_epoch_fn = self.run_epoch_grad_proj if self.pca_grad else self.run_epoch
		self.metrics = defaultdict(list)
		normal_train_iters = kwargs['normal_train_iters']
		best_val_loss = float('inf')
		saved_info = self.model_exists(model, dataset, kwargs)
		if saved_info[0] is not None:
			return saved_info[0], saved_info[1]
		iterator = tqdm(range(self.train_epochs))
		chkpt_path = kwargs["model_chkpt_path"]
		best_path = None
		epochs_to_avg = []
		for i in iterator:
			self.roc_calc = [[], []]
			data_type = kwargs['trainset_id']
			if self.do_multitask and (kwargs['not_is_ft']):
				data_type = "{}.{}".format(kwargs['trainset_id'], kwargs['valset_id'])
			tr_iter = dataset.get_iter(
										kwargs['batch_sz'], type_=data_type,
										tgt_labels_only=kwargs['tgt_labels_only']
									)
			grp = 'tgt' if not kwargs['not_is_ft'] else 'src'
			grp = 'tgt' if kwargs['is_tgt_only'] else grp
			if i >= normal_train_iters:
				tr_loss, tr_acc = train_epoch_fn(model, tr_iter, optim, group=grp)
			else:
				# If required We run a few iterations of training directly
				# on the source distribution with no pca
				tr_loss, tr_acc = self.run_epoch(model, tr_iter, optim, group=grp)

			val_iter = dataset.get_iter(
											kwargs['batch_sz'], type_=kwargs['valset_id'],
											tgt_labels_only=kwargs['tgt_labels_only']
										)
			val_loss, val_acc = self.run_epoch(model, val_iter, None, group='tgt')

			self.metrics['val_auc'].append(self.get_auc(reset=True))
			test_iter = dataset.get_iter(
											kwargs['batch_sz'], type_='tgt_test',
											tgt_labels_only=kwargs['tgt_labels_only']
										)

			test_loss, test_acc = self.run_epoch(model, test_iter, None, group='tgt')
			self.metrics['test_auc'].append(self.get_auc(reset=True))
			# Do the iteration for the monitor val
			monitor_iter = dataset.get_iter(
											kwargs['batch_sz'], type_='monitor_val',
											tgt_labels_only=kwargs['tgt_labels_only']
										)

			monitor_loss, monitor_acc = self.run_epoch(model, monitor_iter, None, group='tgt')
			self.metrics['monitor_auc'].append(self.get_auc(reset=True))

			if lr_scheduler is not None:
				lr_scheduler.step(monitor_loss)
			self.metrics['tr_loss'].append(tr_loss)
			self.metrics['tr_acc'].append(tr_acc)
			self.metrics['val_loss'].append(val_loss)
			self.metrics['val_acc'].append(val_acc)
			self.metrics['test_loss'].append(test_loss)
			self.metrics['test_acc'].append(test_acc)
			self.metrics['monitor_loss'].append(monitor_loss)
			self.metrics['monitor_acc'].append(monitor_acc)
			if i < normal_train_iters:
				cur_best_loss = best_val_loss
			else:
				# gather the last self.patience validation accuracies
				cur_best_loss = min(self.metrics['monitor_loss'][normal_train_iters:][-self.patience:])
			# chkpt if it's time to chkpt
			if i % self.chkpt_every == 0:
				epochs_to_avg.append(i)
				if len(epochs_to_avg) > (self.patience / self.chkpt_every):
					# We want to take checkpoints that are self.patience / self.chkpt_every
					# before and after the best epoch
					epochs_to_avg.pop(0)
				this_path = self.get_chkpt_path(i, chkpt_path)
				# Save the current model with is the best
				torch.save(model.state_dict(), this_path)
			if cur_best_loss < best_val_loss:
				best_val_loss = cur_best_loss
				best_path = self.get_chkpt_path('best', chkpt_path)
				# Save the current model with is the best
				torch.save(model.state_dict(), best_path)
				accs = tr_acc, val_acc, monitor_acc, test_acc
				cur_stats = "Tr Acc : {:.3f} | Val Acc :  {:.3f} | Monitor Acc : {:.3f} | Test Acc : {:.3f}".format(*accs)
				losses = cur_stats, tr_loss, val_loss, monitor_loss, test_loss
				cur_stats = "{}\n Tr Los : {:.3f} | Val Los :  {:.3f} | Monitor Los : {:.3f} | Test Loss : {:.3f}".format(*losses)
				aucs = cur_stats, self.metrics['val_auc'][-1][0], \
					self.metrics['monitor_auc'][-1][0], self.metrics['test_auc'][-1][0]
				cur_stats = "{}\n VL AUC : {:.3f} | Mon AUC :  {:.3f} | Test AUC    : {:.3f}".format(*aucs)
				print('Updating Best : \n', cur_stats)
			elif cur_best_loss == best_val_loss:
				continue
			elif i > self.patience:
				iterator.close()
				print('Stopping because no improvement in {} epochs'.format(self.patience))
				break

		if kwargs['use_last']:
			print('Using the LAST checkpoint instead of the BEST')
		else:
			print('Using the BEST checkpoint instead of the LAST')
			best_path = self.get_chkpt_path('best', chkpt_path)
			model.load_state_dict(torch.load(best_path))
		final_path = self.get_chkpt_path('final', chkpt_path)
		torch.save(model.state_dict(), final_path)
		return self.metrics, max(self.metrics['monitor_acc'])
