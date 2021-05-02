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

import matplotlib.pyplot as plt
import numpy as np

classes = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
linestyles = ['solid', 'dotted', 'dashdot', 'dashed', ':']


def vis_loss_acc(metrics, save_path):
	_, ax = plt.subplots(2, 2, figsize=(10, 10))

	# Viz the Loss
	ax[0][0].plot(metrics['tr_loss'], c='r', label='Training Loss')
	ax[0][0].plot(metrics['val_loss'], c='g', label='Validation Loss')
	ax[0][0].plot(metrics['test_loss'], c='b', label='Test Loss')
	ax[0][0].plot(metrics['monitor_loss'], c='y', label='Monitor Loss')
	ax[0][0].set_ylabel('Loss')
	ax[0][0].set_xlabel('Iter')

	# Viz the Accuracies
	ax[0][1].plot(metrics['tr_acc'], c='r', label='Training Accuracies')
	ax[0][1].plot(metrics['val_acc'], c='g', label='Validation Accuracies')
	ax[0][1].plot(metrics['test_acc'], c='b', label='Test Accuracies')
	ax[0][1].plot(metrics['monitor_acc'], c='y', label='Monitor Accuracies')
	ax[0][1].set_ylabel('Acc')
	ax[0][1].set_xlabel('Iter')

	# Viz AUCs
	metrics['val_auc'] = np.array(metrics['val_auc'])
	metrics['test_auc'] = np.array(metrics['test_auc'])
	metrics['monitor_auc'] = np.array(metrics['monitor_auc'])
	ax[1][0].plot(metrics['val_auc'][:, 0], c='g', label='Validation AUC')
	ax[1][0].plot(metrics['test_auc'][:, 0], c='b', label='Test AUC')
	ax[1][0].plot(metrics['monitor_auc'][:, 0], c='y', label='Monitor AUC')
	ax[1][0].set_ylabel('Auc')
	ax[1][0].set_xlabel('Iter')

	for idx_, class_name in enumerate(classes):
		ax[1][1].plot(
						metrics['val_auc'][:, idx_ + 1], c='g',
						linestyle=linestyles[idx_], label='{} Validation AUC'.format(class_name)
					)
		ax[1][1].plot(
						metrics['test_auc'][:, idx_ + 1], c='b',
						linestyle=linestyles[idx_], label='{} Test AUC'.format(class_name)
					)
		ax[1][1].set_ylabel('Auc')
		ax[1][1].set_xlabel('Iter')

	ax[0][0].legend()
	ax[0][1].legend()
	ax[1][0].legend()
	ax[1][1].legend()

	plt.savefig(save_path)
	plt.close()


def vis_delta_norm(metrics, save_path):
	def plot(ax, stat_matrix, set_):
		names = ["Orig-Norm", "New-Norm", "Pos-In-Span-Norm", "Neg-In-Span-Norm"]
		colors = ['r', 'g', 'y', 'b']
		for idx, name in enumerate(names):
			ax.plot(stat_matrix[:, idx], c=colors[idx], label="{}_{}".format(set_, name))
		ax.set_xlabel('Epoch')
		ax.set_ylabel('Relative to Orig-Norm')
		ax.legend()
	# Viz the Loss
	fig, ax = plt.subplots(2, figsize=(15,15))
	tr_stat_matrix = np.array(metrics['tr_grad_norm_frac'])
	tr_stat_matrix = tr_stat_matrix / tr_stat_matrix[:, :1]  # normalize by old-norm
	plot(ax[0], tr_stat_matrix, 'Train')
	mval_stat_matrix = np.array(metrics['m_val_grad_norm_frac'])
	mval_stat_matrix = mval_stat_matrix / mval_stat_matrix[:, :1]  # normalize by old-norm
	plot(ax[1], mval_stat_matrix, 'Monitor-Validation')
	plt.savefig(save_path)
	plt.close()