# import torch
# import torch.nn as nn
# from torch_geometric.nn import GCNConv,GINConv,SAGEConv,GATConv
# import torch.nn.functional as F
# class Generator(nn.Module):
#     def __init__(self,gnn_type,emb_dim,node_fea_dim,num_layer,dropout_ratio) -> None:
#         super().__init__()
#         self.num_layer = num_layer
#         self.drop_ratio = dropout_ratio

#         ###List of message-passing GNN convs

#         if gnn_type == "gin":
#             self.gnns=GIN(node_fea_dim, emb_dim,dropout_ratio)
#         elif gnn_type == "gcn":
#             self.gnns=GCN(node_fea_dim, emb_dim,dropout_ratio)
#         elif gnn_type == "gat":
#             self.gnns.append(GATConv(emb_dim, input_layer=num_layer))
#         elif gnn_type == "graphsage":
#             self.gnns.append(SAGEConv(emb_dim, input_layer=num_layer))
#     def forward(self,data):
#         output=self.gnns(data.x,data.edge_index)
#         return output
 
# class Discriminator(nn.Module):
#     def __init__(self,gnn_type,emb_dim,node_fea_dim,num_layer,dropout_ratio) -> None:
#         super().__init__()
#         self.num_layer = num_layer
#         self.drop_ratio = dropout_ratio

#         if gnn_type == "gin":
#             self.gnns=GIN(node_fea_dim, emb_dim,dropout_ratio)
#         elif gnn_type == "gcn":
#             self.gnns=GCN(node_fea_dim, emb_dim,dropout_ratio)
#         elif gnn_type == "gat":
#             self.gnns.append(GAT(node_fea_dim, emb_dim,num_layer,dropout_ratio))
#         elif gnn_type == "graphsage":
#             self.gnns.append(GraphSAGE(node_fea_dim, emb_dim,num_layer,dropout_ratio))
#     def forward(self,data):
#         output=self.gnns(data.x,data.edge_index)
#         return output
# class GCN(nn.Module):
#     def __init__(self,node_fea_dim, emb_dim,dropout_ratio):
#         super(GCN, self).__init__()
#         self.dropout_ratio=dropout_ratio
#         self.conv1 = GCNConv(node_fea_dim, emb_dim)
#         self.conv2 = GCNConv(emb_dim, emb_dim)
#         self.mlp =nn.Linear(emb_dim,1)
#     def forward(self,x,edge_index):

#         x = self.conv1(x,edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout_ratio,training=self.training)
#         x = self.conv2(x,edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout_ratio,training=self.training)
#         x = self.mlp(x)
#         return torch.sigmoid(x)
#         # return F.log_softmax(x, dim=1)
# class GIN(nn.Module):
#     def __init__(self,node_fea_dim, emb_dim,dropout_ratio):
#         super(GIN, self).__init__()
#         self.dropout_ratio=dropout_ratio

#         self.BN_layers1=nn.BatchNorm1d(emb_dim)
#         self.BN_layers2=nn.BatchNorm1d(emb_dim)
#         self.gin1=GINConv((nn.Sequential(nn.Linear(node_fea_dim, emb_dim))))
#         self.gin2=GINConv((nn.Sequential(nn.Linear(emb_dim, emb_dim))))
#         self.mlp =nn.Linear(emb_dim,1)


#     def forward(self,x,edge_index):

#         x = self.gin1(x,edge_index)
#         x = self.BN_layers1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout_ratio,training=self.training)
#         x = self.gin2(x,edge_index)
#         x = self.BN_layers2(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=self.dropout_ratio,training=self.training)
#         x = self.mlp(x)
#         return torch.sigmoid(x)
# class GAT(nn.Module):
#     def __init__(self, args):
#         super(GAT, self).__init__()
#         self.BN_layers=nn.ModuleList([nn.BatchNorm1d(args.emb_dim) for i in range(args.num_layer)])
#         self.line=nn.ModuleList([nn.Linear(args.emb_dim*2,args.emb_dim) for i in range(args.num_layer)])
#         self.dropout_ratio=args.dropout_ratio
#         self_loop_attr = torch.zeros(9)
#         self_loop_attr[7] = 1 # attribute for self-loop edge
#         # self_loop_attr = self_loop_attr.to(args.device)
#         self.layers=nn.ModuleList([GATConv(in_channels=args.emb_dim, out_channels=int(args.emb_dim/2),heads=2,concat= True,edge_dim=args.edge_fea_dim) for i in range(args.num_layer)])
#         # self.layers=nn.ModuleList([GATEConv(args.emb_dim,args.emb_dim,args.edge_fea_dim,args.edge_fea_dim,heads=2,concat=True,aggregators='mean') for i in range(args.num_layer)])
#         self.num_layer=args.num_layer

#     def forward(self,x,edge_index,edge_attr):

#         for i in range(self.num_layer):
#             x=self.layers[i](x,edge_index,edge_attr)
#             x=self.BN_layers[i](x)
#             # x=self.line[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout_ratio,training=self.training)
#         return x
    
# class GraphSAGE(nn.Module):
#     def __init__(self, args):
#         super(GraphSAGE, self).__init__()
#         # self.BN=nn.BatchNorm1d(args.emb_dim)
#         self.dropout_ratio=args.dropout_ratio
#         self.layers=nn.ModuleList([SAGEConv(args) for i in range(args.num_layer)])
#         # self.layers=nn.ModuleList([SAGEConv(in_channels=args.emb_dim, out_channels=args.emb_dim) for i in range(args.num_layer)])

#     def forward(self,x,edge_index,edge_attr):
#         for layer in self.layers:
#             x=layer(x,edge_index,edge_attr)
#             # x=layer(x,edge_index)
#             # x=self.BN(x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout_ratio,training=self.training)
#         return x
# import os.path as osp

# import torch
# import torch.nn as nn
# from tqdm import tqdm

# from torch_geometric.datasets import Reddit
# from torch_geometric.loader import NeighborSampler
# from torch_geometric.nn import DeepGraphInfomax, SAGEConv

# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Reddit')
# dataset = Reddit(path)
# data = dataset[0]

# train_loader = NeighborSampler(data.edge_index, node_idx=None,
#                                sizes=[10, 10, 25], batch_size=256,
#                                shuffle=True, num_workers=2)

# test_loader = NeighborSampler(data.edge_index, node_idx=None,
#                               sizes=[10, 10, 25], batch_size=256,
#                               shuffle=False, num_workers=2)


# class Encoder(nn.Module):
#     def __init__(self, in_channels, hidden_channels):
#         super().__init__()
#         self.convs = torch.nn.ModuleList([
#             SAGEConv(in_channels, hidden_channels),
#             SAGEConv(hidden_channels, hidden_channels),
#             SAGEConv(hidden_channels, hidden_channels)
#         ])

#         self.activations = torch.nn.ModuleList()
#         self.activations.extend([
#             nn.PReLU(hidden_channels),
#             nn.PReLU(hidden_channels),
#             nn.PReLU(hidden_channels)
#         ])

#     def forward(self, x, adjs):
#         for i, (edge_index, _, size) in enumerate(adjs):
#             x_target = x[:size[1]]  # Target nodes are always placed first.
#             x = self.convs[i]((x, x_target), edge_index)
#             x = self.activations[i](x)
#         return x


# def corruption(x, edge_index):
#     return x[torch.randperm(x.size(0))], edge_index


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = DeepGraphInfomax(
#     hidden_channels=512, encoder=Encoder(dataset.num_features, 512),
#     summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
#     corruption=corruption).to(device)

# model = model.to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# x, y = data.x.to(device), data.y.to(device)


# def train(epoch):
#     model.train()

#     total_loss = total_examples = 0
#     for batch_size, n_id, adjs in tqdm(train_loader,
#                                        desc=f'Epoch {epoch:02d}'):
#         # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
#         adjs = [adj.to(device) for adj in adjs]
#         print('adjs',adjs)
        
#         optimizer.zero_grad()
#         pos_z, neg_z, summary = model(x[n_id], adjs)
#         loss = model.loss(pos_z, neg_z, summary)
#         loss.backward()
#         optimizer.step()
#         total_loss += float(loss) * pos_z.size(0)
#         total_examples += pos_z.size(0)

#     return total_loss / total_examples


# @torch.no_grad()
# def test():
#     model.eval()

#     zs = []
#     for i, (batch_size, n_id, adjs) in enumerate(test_loader):
#         adjs = [adj.to(device) for adj in adjs]
#         zs.append(model(x[n_id], adjs)[0])
#     z = torch.cat(zs, dim=0)
#     train_val_mask = data.train_mask | data.val_mask
#     acc = model.test(z[train_val_mask], y[train_val_mask], z[data.test_mask],
#                      y[data.test_mask], max_iter=10000)
#     return acc

# if __name__ == '__main__':
#     for epoch in range(1, 31):
#         loss = train(epoch)
#         print(f'Epoch {epoch:02d}, Loss: {loss:.4f}')

#     test_acc = test()
#     print(f'Test Accuracy: {test_acc:.4f}')
# from torch_geometric.datasets import Planetoid
# from torch_geometric.utils import dropout_adj
# dataset = Planetoid(root='./planetoid-master/data', name='Cora')
# import torch
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv

# class GCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = GCNConv(dataset.num_node_features, 16)
#         self.conv2 = GCNConv(16, dataset.num_classes)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index
#         edge_index, _ = dropout_adj(edge_index,p=0.9)
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)

#         return F.log_softmax(x, dim=1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = GCN().to(device)
# data = dataset[0].to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# model.train()
# for epoch in range(200):
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
#     loss.backward()
#     optimizer.step()
# model.eval()
# pred = model(data).argmax(dim=1)
# correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
# acc = int(correct) / int(data.test_mask.sum())
# print(f'Accuracy: {acc:.4f}')        
# train a dominant detector
# from pygod.models import DOMINANT
# from utils import *
# data,split_idx=load_data('./','wiki',10)
#   # hyperparameters can be set here
# for id, (train_idx, test_idx) in enumerate(split_idx):
#     model = DOMINANT(num_layers=4, epoch=20)
#     model.fit(data)  # data is a Pytorch Geometric data object

#     # get outlier scores on the input data
#     outlier_scores = model.decision_scores # raw outlier scores on the input data
#     print(outlier_scores)
#     # predict on the new data in the inductive setting
#     outlier_scores = model.decision_function(data) # raw outlier scores on the input data  # predict raw outlier scores on test
#     print(outlier_scores)
import sys
import numpy as np
from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
data = Dataset(root='./temp', name='cora')
adj, features, labels = data.adj, data.features, data.labels
print(features.shape)
print('adj',adj,adj.shape)
# sys.exit()
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_unlabeled = np.union1d(idx_val, idx_test)
idx_unlabeled = np.union1d(idx_val, idx_test)
# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device='cuda').to('cuda')
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
# Setup Attack Model
model = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
        attack_structure=True, attack_features=False, device='cuda', lambda_=0).to('cuda')
# Attack
model.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=10, ll_constraint=False)
modified_adj = model.modified_adj # modified_adj is a torch.tensor
print('modified_adj',modified_adj,modified_adj.shape)

from deeprobust.graph.data import Dataset
from deeprobust.graph.defense import GCN
from deeprobust.graph.targeted_attack import Nettack
data = Dataset(root='/tmp/', name='cora')
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
# Setup Surrogate model
surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
# Setup Attack Model
target_node = 0
model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device='cpu').to('cpu')
# Attack
model.attack(features, adj, labels, target_node, n_perturbations=5)
modified_adj = model.modified_adj # scipy sparse matrix
modified_features = model.modified_features # scipy sparse matrix