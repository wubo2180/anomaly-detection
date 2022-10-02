import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import InMemoryDataset, download_url
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_data(root,datasets,num_folds):
    # load the adjacency
    adj = np.loadtxt(root+'/data/'+datasets+'.txt')
    num_user = len(set(adj[:, 0]))
    num_object = len(set(adj[:, 1]))
    adj = adj.astype('int')
    nb_nodes = np.max(adj) + 1
    edge_index = adj.T
    print('Load the edge_index done!')
    # print('adj',adj)
    # load the user label
    label = np.loadtxt(root+'/data/'+datasets+'_label.txt')
    y = label[:, 1]
    print('Ratio of fraudsters: ', np.sum(y) / len(y))
    print('Number of edges: ', edge_index.shape[1])
    print('Number of users: ', num_user)
    print('Number of objects: ', num_object)
    print('Number of nodes: ', nb_nodes)
    
    # split the train_set and validation_set

    split_idx = []
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
    for (train_idx, test_idx) in skf.split(y, y):
        # print(test_idx)
        # print(test_idx.shape)
        split_idx.append((train_idx, test_idx))
    # load initial features
    feats = np.load(root+'/features/'+datasets+'_feature64.npy')
    # print('feats',feats,feats.shape)
    edge_index=torch.from_numpy(edge_index).long()
    feats=torch.from_numpy(feats).float()
    label=torch.from_numpy(label).float()
    data=Data(x=feats,edge_index=edge_index,y=label[:,1].reshape(-1,1),nb_nodes=nb_nodes,label=label.long())
    # print(max(edge_index[0]))
    # print(max(edge_index[1]))
    return data,split_idx,nb_nodes

def plot_xy(x_values, label, title,name):
    """绘图"""
    df = pd.DataFrame(x_values, columns=['x', 'y'])
    df['label'] = label
    sns.scatterplot(x="x", y="y", hue="label", data=df,sizes=(40, 40))
    # plt.title(title)
    plt.xlabel("",fontsize=40)
    plt.ylabel("",fontsize=40)
    plt.legend(['Normal','Abnormal'],fontsize=15)
    plt.savefig(name+'.pdf')
    plt.show()


def plot_tsne(x_value, y_value,name):
    # x_value, y_value = get_data()
    # PCA 降维
    
    # pca = PCA(n_components=2)
    # x_pca = pca.fit_transform(x_value)
    # plot_xy(x_pca, y_value, "PCA")
    # t-sne 降维
    
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(x_value)
    plot_xy(x_tsne, y_value, "t-sne",name)