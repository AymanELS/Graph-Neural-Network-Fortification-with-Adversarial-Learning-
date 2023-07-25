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
parser.add_argument('--input_dim', type=int, default=64, help='size of input dimension of generator')
parser.add_argument('--hidden_dim', type=int, default=128, help='size of hidden dimension of generator')
parser.add_argument('--output_dim', type=int, default=30, help='size of output of generator')
parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--test_size', type=int, default=500, help='size of testing sample')
parser.add_argument('--noise_level', type=int, default=3, help='degree of noise to add to graphs')
parser.add_argument('--outdir', default='adversarial_results', help='output directory')

opt = parser.parse_args()

#load benign graphs
seed = 0

input_dim = opt.input_dim #graph embedding size
hidden_dim = opt.hidden_dim
output_dim = opt.output_dim
noise_level = opt.noise_level
epochs = opt.epochs
batch_size = opt.batch_size
test_size = opt.test_size
noise_size = output_dim
outdir = opt.outdir
dataset = opt.dataset
if not os.path.exists(outdir):
	os.mkdir(outdir)

def get_pred_int(pred_score):
	if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
		return (pred_score >  0.5).long()
	else:
		return pred_score.max(dim=1)[1]

def load_malwareapi():
	df = pd.read_csv('MalwareAPISequence.csv')
	benign = df.loc[df['malware']==0]
	malicious = df.loc[df['malware']==1]
	ben = benign.sample(n=1079, random_state=seed) # n=batch_size
	mal = malicious.sample(n=1079, random_state=seed) # n=batch_size
	real_ben_y = ben['malware'].values.astype(int)
	real_mal_y = mal['malware'].values.astype(int)
	real_ben_calls = ben.drop(['hash', 'malware'], axis = 1)
	real_mal_calls = mal.drop(['hash', 'malware'], axis = 1)
	return real_ben_calls, real_mal_calls, real_ben_y, real_mal_y

def toNXGraph(calls, y):
	graphs = []
	#print(len(calls))
	calls = calls.values.astype(int)
	for i in range(len(calls)):
		#print(i)
		#G = nx.DiGraph()
		G = nx.Graph()
		for j in range(len(calls[0])-1):
			G.add_edge(calls[i][j], calls[i][j+1])
		for j in list(G.nodes):
			G.nodes[j]["node_feature"]=torch.tensor([j])
		G.graph["graph_label"] = torch.tensor([y[i]]) 
		graphs.append(G)
	return graphs

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


def create_adv_graph(graphs, noise, attacker_output):
	# print(graphs[1].edges)
	#print(noise)
	#print(torch.argmax(attacker_output[0]).item())
	for i in range(len(graphs)):
		# print(torch.argmax(attacker_output[i]).item())
		selected_noise = noise[torch.argmax(attacker_output[i]).item()]
		#print(selected_noise)
		graphs[i].add_edges_from(selected_noise)
		for j in list(graphs[i].nodes):
			graphs[i].nodes[j]["node_feature"]=torch.tensor([j])
	return graphs

def noise(benign_graphs, level=5, size = 20):
	benign_noise=[]
	for i in range(size):
		graph = choice(benign_graphs) #make random
		while len(graph.edges) <= level:
			graph = choice(benign_graphs)
		sample_edges = sample(graph.edges(), level)
		benign_noise.append(sample_edges)
	return benign_noise


class Attacker(nn.Module):
	def __init__(self):
		super(Attacker, self).__init__()
		self.attacker = nn.Sequential(
			nn.Linear(in_features=input_dim, out_features=hidden_dim),
			nn.LeakyReLU(),
			nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
			nn.LeakyReLU(),
			nn.Linear(in_features=hidden_dim, out_features=noise_size),
			nn.Softmax(dim=1)
			)
	def forward(self, x):
		output = self.attacker(x)
		return output

def init_weights(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform(m.weight)
		m.bias.data.fill_(0.01)


def label_smoothing(label):
	for i in range(len(label)):
		if label[i]==0:
			label[i]=random.uniform(0.0, 0.5)
		if label[i]==1:
			label[i]=random.uniform(0.5, 1)


if dataset == 'malwareapi':
	if os.path.exists(os.path.join(outdir, 'malwareapi_benign_graphs.pkl')):
		with open(os.path.join(outdir,'malwareapi_benign_graphs.pkl'), 'rb') as f:
			benign_graphs = pkl.load(f)
		with open(os.path.join(outdir,'malwareapi_malware_graphs.pkl'), 'rb') as f:
			malware_graphs = pkl.load(f)
	else:
		real_ben_calls, real_mal_calls, real_ben_y, real_mal_y = load_malwareapi()
		benign_graphs = toNXGraph(real_ben_calls, real_ben_y)
		malware_graphs = toNXGraph(real_mal_calls, real_mal_y)
		# save benign and malware graphs
		with open(os.path.join(outdir,'malwareapi_benign_graphs.pkl'), 'wb') as f:
			pkl.dump(benign_graphs, f)
		with open(os.path.join(outdir,'malwareapi_malware_graphs.pkl'), 'wb') as f:
			pkl.dump(malware_graphs, f)
elif dataset == 'CICandroid':
	pass
else:
	print('Dataset not defined')


## initialize models and dataloader
attacker = Attacker()
attacker.train()
attacker.apply(init_weights)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(attacker.parameters(), lr=0.0001, betas=(0.9, 0.999))
mal_loader = dataset_loader(malware_graphs)
GNN_model = torch.load('model.pt')
GNN_model.eval()
n=1079
mini_batch=int(n/batch_size)

def train():
	with open(os.path.join(outdir,'malwareapi_malware_graphs.pkl'), 'rb') as f:
		mal_graphs = pkl.load(f)
	min_acc2=float('inf')
	for epoch in range(epochs):
		for i in range(mini_batch+1):
			if i==mini_batch:
				sample_size= n-(batch_size*mini_batch)
			else:
				sample_size=batch_size
			attacker.zero_grad()
			pred, true, graph_emb = GNN_model(next(iter(mal_loader)))
			attacker_output = attacker(graph_emb)
			# pred_int = get_pred_int(pred)
			# acc1 = accuracy_score(true, pred_int)
			# malware_graphs = toNXGraph(real_mal_calls, real_mal_y)[:batch_size]
			malware_graphs = mal_graphs[sample_size*i: sample_size*(i+1)]
			# print(len(malware_graphs))
			benign_noise = noise(benign_graphs, level=noise_level, size = noise_size)
			adv_malware_graphs = create_adv_graph(malware_graphs, benign_noise, attacker_output)
			adv_mal_loader = dataset_loader(adv_malware_graphs, batch_size=batch_size)
			pred2, true2, _ = GNN_model(next(iter(adv_mal_loader)))
			pred_int2 = get_pred_int(pred2)
			acc2 = accuracy_score(true2, pred_int2)
			label = torch.zeros(pred2.shape[0], 1)
			label_smoothing(pred2)
			label_smoothing(label)
			loss = criterion(pred2, label)
			loss.backward()
			optimizer.step()
			if acc2 < min_acc2:
				min_acc2=acc2
				best_attacker_state = attacker.state_dict()
				best_noise = benign_noise
			print('[%d/%d][%d/%d] loss: %.4f, acc2: %.4f' % (epoch, epochs, i, mini_batch,loss.item(), acc2.item()))
	return best_attacker_state, best_noise


def test_attack_model(attacker, noise, noise_level):
	with open(os.path.join(outdir,f'{dataset}_malware_graphs.pkl'), 'rb') as f:
		malware_graphs = pkl.load(f)
	mal_loader = dataset_loader(malware_graphs, batch_size=test_size)
	pred, true, graph_emb = GNN_model(next(iter(mal_loader)))
	pred_int = get_pred_int(pred)
	acc1 = accuracy_score(true, pred_int)
	f1_score1 = f1_score(true, pred_int)
	attacker_output = attacker(graph_emb)
	# malware_graphs = toNXGraph(real_mal_calls, real_mal_y)[:test_size]
	with open(os.path.join(outdir,f'{dataset}_malware_graphs.pkl'), 'rb') as f:
		malware_graphs = pkl.load(f)[:test_size]
	adv_malware_graphs = create_adv_graph(malware_graphs, noise, attacker_output)
	adv_mal_loader = dataset_loader(adv_malware_graphs, batch_size=test_size)
	pred2, true2, _ = GNN_model(next(iter(adv_mal_loader)))
	pred2_int = get_pred_int(pred2)
	acc2 = accuracy_score(true2, pred2_int)
	f1_score2 = f1_score(true2, pred2_int)
	with open(os.path.join(outdir, f'{dataset}_adv_only_noise{noise_level}.pkl'), 'wb') as f:
		pkl.dump(adv_malware_graphs, f)
	return adv_malware_graphs, acc1, acc2, f1_score1, f1_score2


best_attacker_state, best_noise = train()
torch.save(best_attacker_state, os.path.join(outdir, f'best_attacker{noise_level}.pt'))
with open(os.path.join(outdir, f'best_noise{noise_level}.pkl'), 'wb') as f:
	pkl.dump(best_noise, f)
attacker.load_state_dict(best_attacker_state)
adv_graphs, acc1, acc2, f1_score1, f1_score2 = test_attack_model(attacker, best_noise, noise_level)



def augment_dataset(graphs, adv_graphs, noise_level, dataset_name):
	new_graphs = adv_graphs + graphs
	with open(os.path.join(outdir, f'{dataset_name}_augmented_noise{noise_level}.pkl'), 'wb') as f:
		pkl.dump(new_graphs, f)
	with open(f'GraphGym-master/run/datasets/{dataset_name}_augmented_noise{noise_level}.pkl', 'wb') as f:
		pkl.dump(new_graphs, f)

with open('malwareapi_balanced.pkl', 'rb') as f:
	graphs = pkl.load(f)
augment_dataset(graphs, adv_graphs, noise_level, dataset)


print(f'Performance of GNN on original malware: Accuracy {acc1}, F1_score {f1_score1}')
print(f'Performance of GNN on adversarial malware: Accuracy {acc2}, F1_score {f1_score2}')
