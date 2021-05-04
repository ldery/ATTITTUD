# Auxiliary Task Update Decomposition : The Good, The Bad and The Neutral

This repository contains the source code for the paper [Auxiliary Task Update Decomposition : The Good, The Bad and The Neutral](https://openreview.net/forum?id=1GTma8HwlYp), by Lucio M Dery, Yann Dauphin, David Grangier, ICLR 2021.

---

<p align="center"> 
    <img src="https://github.com/ldery/ATTITTUD/blob/main/g_aux_11-1.png" width="800">
</p>

## Links

1. [Paper](https://openreview.net/forum?id=1GTma8HwlYp)

## Installation

1. conda env create --file attitud.yml
2. Download Tiny Imagenet from : https://www.kaggle.com/c/tiny-imagenet/data
	2a. Change IMGNET_PATH at the top of data/dataset.py to desired location
3. Request ChexPert v1-small from : https://stanfordmlgroup.github.io/competitions/chexpert/
	3a. Change CHXPERT_PATH at the top of data/dataset.py to desired location

## Running

### To obtain results on ChexPert Dataset

#### Baseline - Initialized with Random Resnet
python -u main.py -train-perclass 1000 -num-monitor 100  -imgnet-per-class 250 -imgnet-n-classes  200  -dataset-type CHEXPERT -model-type ResNet -num-runs 5 -no-src-only -src-batch-sz 128 -tgt-batch-sz 64 -patience 10 -train-epochs 100 -use-last-chkpt -is-chexnet  -dropRate 0.3 -lr 1e-3 -base-resnet '18' -exp-name CHEXPERT/pretrained


#### Baseline - Initialized with Pre-trained Resnet 
python -u main.py -pretrained -train-perclass 1000 -num-monitor 100  -imgnet-per-class 250 -imgnet-n-classes  200  -dataset-type CHEXPERT -model-type ResNet -num-runs 5 -no-src-only -src-batch-sz 128 -tgt-batch-sz 64 -patience 10 -train-epochs 100 -use-last-chkpt -is-chexnet  -dropRate 0.3 -lr 1e-3 -base-resnet '18' -exp-name CHEXPERT/randomInit


##### Ours - Auxiliary Task Update Decomposition
python -u main.py -pretrained -train-perclass 1000 -num-monitor 100 -imgnet-per-class 500 -imgnet-n-classes 200 -dataset-type CHEXPERT -model-type ResNet -num-runs 5 -no-tgt-only -no-src-only -src-batch-sz 128 -tgt-batch-sz 64 -patience 10 -train-epochs 100 -use-last-chkpt -is-chexnet -ft-model src_pca -num-pca-basis 10 -pca-every 10 -use-jvp -pca-nsamples 64 -lowrank-nsamples 32 -pca-grad -ft-dropRate 0.2 -dropRate 0.2 -proj_lambda "(1.0, 1.0, -1.0)" -lr 1e-4 -finetune-lr 5e-5 -base-resnet '18' -exp-name CHEXPERT/attittud
