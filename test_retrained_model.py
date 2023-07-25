import os
import time
import numpy
import torch
import random
import argparse
import pandas as pd
import pickle as pkl
import networkx as nx
import torch.nn as nn
import torch.optim as optim
from deepsnap.batch import Batch
from random import choice, sample
from deepsnap.dataset import GraphDataset
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from graphgym.loader import transform_before_split, transform_after_split


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='malwareapi', help='dataset path')
parser.add_argument('--test_size', type=int, default=500, help='size of testing sample')
parser.add_argument('--attacking_noise_level', type=int, default=5, help='degree of noise used to retrain GNN model')
parser.add_argument('--testing_noise_level', type=int, default=3, help='degree of noise used to test GNN model')
parser.add_argument('--outdir', default='adversarial_results', help='output directory')
opt = parser.parse_args()



#load benign graphs

dataset=opt.dataset
attacking_noise_level = opt.attacking_noise_level
testing_noise_level = opt.testing_noise_level
test_size = opt.test_size
outdir = opt.outdir



def get_pred_int(pred_score):
	if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
		return (pred_score >  0.5).long()
	else:
		return pred_score.max(dim=1)[1]

def dataset_loader(graphs, batch_size=64, shuffle=False):
	dataset = GraphDataset(
		graphs,
		task='graph',
		edge_train_mode='all',
		edge_message_ratio=0.8,
		edge_negative_sampling_ratio=1.0,
		resample_disjoint=False,
		minimum_node_per_graph=0)

	dataset = transform_before_split(dataset)
	# datasets = dataset.split(transductive=cfg.dataset.transductive,
	# 							 split_ratio=cfg.dataset.split,
	# 							 shuffle=cfg.dataset.shuffle_split)
	
	datasets = transform_after_split([dataset])
	loader = DataLoader(datasets[0],
							collate_fn=Batch.collate(),
							batch_size=batch_size,
							shuffle=False,
							num_workers=1,
							pin_memory=False)
	return loader




#command = 'sbatch GraphGym-master/run/GNN_script_adv.sbatch'

# with open(os.path.join(outdir,f'{dataset}_augmented_noise{noise_level}.pkl'), 'rb') as f:
# 	graphs_augmented = pkl.load(f)
with open(os.path.join(outdir,f'{dataset}_adv_only_noise{testing_noise_level}.pkl'), 'rb') as f:
	graphs_augmented = pkl.load(f)

GNN_model_retrained = torch.load(os.path.join(outdir, f'model_retrained{attacking_noise_level}.pt'))
GNN_model_retrained.eval()
graph_loader = dataset_loader(graphs_augmented, batch_size=test_size)
pred, true, _ = GNN_model_retrained(next(iter(graph_loader)))
pred_int = get_pred_int(pred)
new_acc = accuracy_score(true, pred_int)
new_f1score = accuracy_score(true, pred_int)


print(f'Performance of retrained GNN on adversarial malware: Accuracy {new_acc}, F1_score {new_f1score}')
