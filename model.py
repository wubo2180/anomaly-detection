import torch
import torch.nn as nn
# from torch.distributions.bernoulli import Bernoulli
from torch_geometric.nn import GCNConv,GINConv,SAGEConv,GATConv,DeepGraphInfomax,Linear
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj,dense_to_sparse,to_dense_adj
import sys
class Generator(nn.Module):
    def __init__(self,gnn_type,emb_dim,node_fea_dim,dropout_ratio,num_layer,nb_nodes) -> None:
        super().__init__()
        
        self.alpha=0.5
        self.dropout_ratio = dropout_ratio
        self.num_layer = num_layer
        self.batch_norm = nn.ModuleList()
        self.gnn = nn.ModuleList()

        self.adj_changes = nn.Parameter(torch.FloatTensor(nb_nodes,nb_nodes))
        self.adj_changes.data.fill_(0)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.feature_change = nn.Parameter(torch.FloatTensor(node_fea_dim,node_fea_dim))
        self.feature_change.data.fill_(1.0)
        self.feature_transform=nn.Linear(node_fea_dim,emb_dim)
        for i in range(num_layer):
            if gnn_type == 'gin':
                self.gnn.append(GINConv((nn.Sequential(nn.Linear(emb_dim, emb_dim)))))
            elif gnn_type == 'gcn':
                self.gnn.append(GCNConv(emb_dim,emb_dim))
            elif gnn_type == 'gat':
                self.gnn.append(GATConv(emb_dim,emb_dim))
            elif gnn_type == 'graphsage':
                self.gnn.append(SAGEConv(emb_dim,emb_dim))
            # self.gnn.append(GATConv(node_fea_dim,emb_dim))
            elif gnn_type == 'DGI':
                DeepGraphInfomax(hidden_channels=emb_dim, encoder=Encoder(emb_dim, emb_dim),
                                summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
                                corruption=corruption)
            else:
                raise ValueError("Gnn type error!")
            self.batch_norm.append(nn.BatchNorm1d(emb_dim))
        self.mlp = nn.Linear(emb_dim,1)
    
    def forward(self,data,latent):
        
        x, edge_index = data.x,data.edge_index
        
        adj = to_dense_adj(data.edge_index).squeeze()
        # print(adj)
        adj =adj.to_sparse()
        modified_adj =self.get_modified_adj(adj)
        # adj = self.sigmoid(adj)
        # adj = torch.bernoulli(adj).long()
        edge_index , edge_attr = dense_to_sparse(modified_adj)
        x_ = x
        x = x @ self.feature_change
        x = self.feature_transform(x)
        # x = self.relu(x)
        for layer in range(self.num_layer):
            x = self.gnn[layer](x,edge_index)
            x = self.batch_norm[layer](x)
            x = self.dropout(x)
            x = self.relu(x)
        x = self.mlp(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        score = self.loss_func(x,x_,modified_adj,adj)
        # print(score)
        # sys.exit()
        # score=1.

        return x,torch.mean(score)
        # return x,score

    def loss_func(self, x, x_, s, s_):
        # attribute reconstruction loss
        # diff_attribute = torch.pow(torch.mean(x,0) - torch.mean(x_,0), 2)
        diff_attribute = torch.pow(x-x_, 2)
        # print(diff_attribute)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))
        # attribute_errors = torch.sum(torch.sqrt(diff_attribute))

        # structure reconstruction loss
        diff_structure = torch.pow(s - s_, 2)
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

        score = self.alpha * attribute_errors \
                + (1 - self.alpha) * structure_errors
        return score
    def get_modified_adj(self, ori_adj):
        # print(ori_adj.to_dense().sum())
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        # print('adj_changes_square',adj_changes_square,adj_changes_square.shape)
        # ind = np.diag_indices(self.adj_changes.shape[0]) # this line seems useless
        # print('ori_adj',ori_adj,adj_changes_square)
        # if self.undirected:
        #     adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
        # print('adj_changes_square',adj_changes_square,adj_changes_square.shape)
        modified_adj = adj_changes_square + ori_adj
        # print('modified_adj',modified_adj,modified_adj.shape)
        # print(modified_adj.sum())
        return modified_adj


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            GINConv(in_channels, hidden_channels),
            GINConv(hidden_channels, hidden_channels)
        ])

        self.activations = torch.nn.ModuleList()
        self.activations.extend([
            nn.ReLU(hidden_channels),
            nn.ReLU(hidden_channels)
        ])

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = self.activations[i](x)
        return x


def corruption(x, edge_index):
    return x[torch.randperm(x.size(0))], edge_index


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Discriminator(nn.Module):
    def __init__(self,gnn_type,emb_dim,node_fea_dim,dropout_ratio,num_layer) -> None:
        super().__init__()
        self.dropout_ratio=dropout_ratio
        # self.embedding = None
        self.num_layer=num_layer
        # print('&&&&', num_layer,dropout_ratio)
        self.batch_norm=nn.ModuleList()
        self.gnn=nn.ModuleList()
        self.feature_transform=nn.Linear(node_fea_dim,emb_dim)
        self.dropout=nn.Dropout(p=dropout_ratio)
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        for i in range(num_layer):
            if gnn_type=='gin':
                self.gnn.append(GINConv((nn.Sequential(nn.Linear(emb_dim, emb_dim)))))
            elif gnn_type=='gcn':
                self.gnn.append(GCNConv(emb_dim,emb_dim))
            elif gnn_type=='gat':
                self.gnn.append(GATConv(emb_dim,emb_dim))
            elif gnn_type=='graphsage':
                self.gnn.append(SAGEConv(emb_dim,emb_dim))
            # self.gnn.append(GATConv(node_fea_dim,emb_dim))
            elif gnn_type=='dgi':
                DeepGraphInfomax(hidden_channels=emb_dim, encoder=Encoder(emb_dim, emb_dim),
                                summary=lambda z, *args, **kwargs: torch.sigmoid(z.mean(dim=0)),
                                corruption=corruption)
            else:
                raise ValueError("Gnn type error!")
            self.batch_norm.append(nn.BatchNorm1d(emb_dim))
        self.mlp =nn.Linear(emb_dim,1)
    
    def forward(self,data):
        x,edge_index = data.x,data.edge_index
        # edge_index, _ = dropout_adj(edge_index,p=0.9)
        # print(data.edge_index)
        x = self.feature_transform(x)
        x = self.relu(x)
        for layer in range(self.num_layer):
            x = self.gnn[layer](x,edge_index)
            x = self.batch_norm[layer](x)
            x = self.dropout(x)
            # print(x)
            # break
            x = self.relu(x)
        embedding = x
        x = self.mlp(x)
        # print(self.dropout_ratio)
        x = self.dropout(x)
        x = self.sigmoid(x)
        # print(x)
        return x,embedding
