import argparse
import random
# from sklearn import metrics

from sklearn.metrics import roc_auc_score
from torch_geometric.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import sys
import time
from progressbar import *
from utils import *
from model import *
import os,sys
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dropout_adj,to_dense_adj,dense_to_sparse,is_undirected
# from torch_geometric.nn import DataLoader
# from torch_geometric.data import DataLoader

def evaluate(model, data,test_idx):
    model.eval()
    
    pred,_  =  model(data)
    pred  =  pred[data.label[test_idx,0]].detach().cpu().numpy()
    target  =  data.y[test_idx].detach().cpu().numpy()
    auc  =  roc_auc_score(target,pred)

    return auc


def train(args,net,data,train_idx,test_idx,criterion,optimizer):
    test_auc = []

    loss_list = []

    for epoch in tqdm(range(1,args.epochs + 1),ncols=50):
        net.train()
        optimizer.zero_grad()
        output,embedding = net(data)
        loss = criterion(output[data.label[train_idx,0]],data.y[train_idx])
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        test_auc.append(evaluate(net,data,test_idx))

    return np.max(test_auc),loss_list,embedding.detach().cpu().numpy()

def main(args):
    data,split_idx,nb_nodes = load_data(args.dir,args.dataset,args.num_folds)
    label=data.label.numpy()
    index=np.random.randint(0,label.shape[0],size=(min(label.shape[0],1000),))
    # print(index.shape)
    # plot_tsne(data.x.numpy()[label[:,0]][index],label[:,1][index],name='original')
    args.node_fea_dim = data.x.shape[1]

    print(data.edge_index.shape)
    new_edge=torch.from_numpy(np.random.choice(nb_nodes,(2,1000)))
    # torch.from_numpy(new_edge)
    # edge_index = torch.tensor([[0, 1, 1, 2],
    #                        [1, 0, 2, 1]], dtype=torch.long)
    # print(is_undirected(edge_index))
    # print(is_undirected(data.edge_index))
    
    data.edge_index = torch.cat((data.edge_index,new_edge),1)
    print(data.edge_index.shape)
    # data.edge_index, _ = dropout_adj(data.edge_index,p=0.5)
    data = data.to(args.device)
    test_auc_list = []
    for id, (train_idx, test_idx) in enumerate(split_idx):
        criterion = nn.BCELoss()
        train_idx = torch.from_numpy(train_idx).long().to(args.device)
        test_idx = torch.from_numpy(test_idx).long().to(args.device)
        net = Discriminator(args.gnn_type,args.emb_dim,args.node_fea_dim,args.dropout_ratio,args.num_layer).to(args.device)

        optimizer = optim.Adam(net.parameters(), lr = args.lr, weight_decay = args.decay)

        test_auc,loss_list,embedding = train(args,net,data,train_idx,test_idx,criterion,optimizer)

        print('test_auc: {:.4f}'.format(test_auc))
        test_auc_list.append(test_auc)
        # break
    print('mean test_auc: {:.4f}'.format(np.mean(test_auc_list)))
    # print('loss',loss_list[2])
    # label=data.label.cpu().numpy()
    # index=np.random.randint(0,label.shape[0]-1,size=(1000,))
    # print(embedding[label[:,0]][index],label[:,1][index])
    # plot_tsne(embedding[label[:,0]][index],label[:,1][index],name='trained')

if __name__  ==  "__main__":
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    parser = argparse.ArgumentParser(
        description = 'PyTorch implementation')
    parser.add_argument('--device', type = int, default = 1,
                        help = 'which gpu to use if any (default: 0)')
    parser.add_argument('--dataset', type = str, default = 'amazon',
                        help = 'dataset name (wiki, reddit, alpha,amazon)')
    parser.add_argument('--dir', type = str, default = '.',
                        help = 'dataset directory')
    parser.add_argument('--delta', type = float, default = 0.1,
                        help = 'dataset directory')    
    parser.add_argument('--epochs', type = int, default = 300,
                        help = 'number of epochs to train (default: 100)')
    parser.add_argument('--decay', type = float, default = 0,
                        help = 'weight decay (default: 0)')
    parser.add_argument('--dropout_ratio', type = float, default = 0.5,
                        help = 'dropout ratio (default: 0)')
    parser.add_argument('--emb_dim', type = int, default = 128,
                        help = 'embedding dimensions (default: 128)')
    parser.add_argument('--gnn_type', type = str, default = "gin",
                        help = 'gin type (gin,gcn,graphsage,gat)')
    parser.add_argument('--lr', type = float, default = 0.01,
                        help = 'learning rate (default: 0.01)')
    parser.add_argument('--num_workers', type = int, default = 0, help = 'number of workers for dataset loading')
    parser.add_argument('--num_layer', type = int, default = 2,
                        help = 'number of GNN message passing layers (default: 2).')
    parser.add_argument('--node_fea_dim', type = int, default = 64,
                        help = 'node feature dimensions (BIO: 2; DBLP: 10; CHEM: ))')
    parser.add_argument('--num_folds', type = int, default = 10,
                        help = 'number of folds (default: 10)')
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    args.device = device
    print(args)
    main(args)
