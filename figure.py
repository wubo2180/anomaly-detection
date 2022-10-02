from matplotlib import rcParams
config = {
    "font.family":'Times New Roman',  # 设置字体类型
    "font.size": 38,
#     "mathtext.fontset":'stix',
}

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator, figure
# plt.style.use("ggplot")

fig=plt.figure(figsize=(12,9))
plt.grid(True,linestyle='-')
num_list = [69.1,69.2]
num_list1 = [68.4,69.5]
num_list2 = [71.1,71.2]

x = np.arange(2) #总共有几组，就设置成几，我们这里有三组，所以设置为3

total_width, n = 0.5, 2 # 有多少个类型，只需更改n即可，比如这里我们对比了四个，那么
my_y_ticks = np.arange(65,75, 1)

plt.ylim((65, 75))

plt.yticks(my_y_ticks,fontsize=30)
plt.yticks(my_y_ticks,fontsize=30)
#就把n设成4
plt.xticks([0,1],['Biology','PreDBLP'],fontsize=30)
width = total_width / n

x = x - (total_width - width) / 2
plt.bar(x, num_list,width=width, label='MVRACE (node)', color='orange',zorder=10)
# for i in range(len(x)):
#     x[i] += width
plt.bar(x+ width, num_list1, width=width, label='MVRACE (graph)', color='lightgreen',zorder=10)
# plt.bar(x, num_list2, width=width, label='RSS-GNN (+SDG)', fc='g')
# for i in range(len(x)):
#     x[i] += width
plt.bar(x+ 2*width, num_list2, width=width, label='MVRACE', color='steelblue',zorder=10)
# plt.bar(x, num_list2, width=width, label='RSS-GNN (mmd)', fc='y')
# p1=plt.plot(x, y111, "--or",label='Micro F1')

plt.xlabel(r"budget $\Delta$",fontsize=30)
plt.ylabel("ROC-AUC (%)",fontsize=30)
plt.legend(fontsize=25)
plt.savefig('./figure1.pdf',bbox_inches='tight')



##dimension
# fig=plt.figure(figsize=(12,9))
# plt.grid(True,linestyle='--')

# GCN=[0.6450,0.6513,0.6441,0.6446,0.6594]
# CARE_GNN=[0.6616,0.6716,0.6697,0.6713,0.6654]
# CONDA = [0.5913,0.5943,0.6121,0.601,0.5999]
# PB_AAD = [0.7214,0.7134,0.7315,0.7216,0.7123]
# x = np.arange(5)
# # plt.xlim([2,10])
# plt.xticks([0,1,2,3,4],['0.1','0.3','0.5','0.7','0.9'],fontsize=30)
# my_y_ticks = np.arange(65, 75, 1)
# plt.ylim((0.5, 0.8))
# my_y_ticks=[0.5,0.6,0.7,0.8]
# plt.yticks(my_y_ticks,fontsize=30)
# plt.ylabel("ROC-AUC (%)",fontsize=30)
# plt.xlabel(r"$\alpha$",fontsize=30)
# plt.plot(x , GCN, color='g',linestyle='--',marker='o',markersize=16,linewidth=4,label='GCN')
# plt.plot(x , CARE_GNN, color='mediumpurple',linestyle='-',marker='d',markersize=16,linewidth=4,label='CARE_GNN')
# plt.plot(x , CONDA, color='g',linestyle='-',marker='o',markersize=16,linewidth=4,label='CONDA')
# plt.plot(x , PB_AAD, color='mediumpurple',linestyle='--',marker='d',markersize=16,linewidth=4,label='PB_AAD')
# plt.legend(fontsize=25)
# plt.savefig('figure2.pdf',bbox_inches='tight')


###belta

fig=plt.figure(figsize=(12,9))
plt.grid(True,linestyle='--')
# ax2_2 = plt.twinx() # 用于绘制双Y轴，重点。
# ax2.grid(True,linestyle='--',zorder=0)
# ax2.set_ylim([0.6, 0.8])
PB_AAD_clean=[0.8713,0.8615,0.8756,0.8812,0.8753]
PB_AAD_random=[0.8219,0.8145,0.8154,0.8216,0.8064]
PB_AAD_MO = [0.7214,0.7315,0.7216,0.7131,0.7065]
# print(len(Biology))
x = np.arange(5)
# my_y_ticks = np.arange(0.6, 0.8, 0.05)
plt.xticks([0,1,2,3,4],['0.1','0.3','0.5','0.7','0.9'],fontsize=30)
my_y_ticks = np.arange(65, 75, 1)
plt.ylim((0.5, 1.0))
my_y_ticks=[0.5,0.6,0.7,0.8,0.9]
plt.yticks(my_y_ticks,fontsize=30)
plt.ylabel("ROC-AUC (%)",fontsize=30)
plt.xlabel(r"$\beta$",fontsize=30)
plt.plot(x , PB_AAD_clean, color='r',linestyle='-',marker='o',markersize=16,linewidth=4,label='Clean')
plt.plot(x , PB_AAD_random, color='black',linestyle='-',marker='d',markersize=16,linewidth=4,label='Random')
plt.plot(x , PB_AAD_MO, color='g',linestyle='-',marker='d',markersize=16,linewidth=4,label='MO-GAA')

plt.legend(fontsize=25)
plt.savefig('figure_belta.pdf',bbox_inches='tight')

fig=plt.figure(figsize=(12,9))
plt.grid(True,linestyle='--')
# ax2_2 = plt.twinx() # 用于绘制双Y轴，重点。
# ax2.grid(True,linestyle='--',zorder=0)
# ax2.set_ylim([0.6, 0.8])
PB_AAD_clean=[0.8713,0.8815,0.8601,0.8912,0.8711]
PB_AAD_random=[0.8219,0.8315,0.8213,0.8021,0.8131]
PB_AAD_MO = [0.7214,0.7015,0.7164,0.7131,0.7134]
# print(len(Biology))
x = np.arange(5)
# my_y_ticks = np.arange(0.6, 0.8, 0.05)
plt.xticks([0,1,2,3,4],['0.1','0.3','0.5','0.7','0.9'],fontsize=30)
my_y_ticks = np.arange(65, 75, 1)
plt.ylim((0.5, 1.0))
my_y_ticks=[0.5,0.6,0.7,0.8,0.9]
plt.yticks(my_y_ticks,fontsize=30)
plt.ylabel("ROC-AUC (%)",fontsize=30)
plt.xlabel(r"$\gamma$",fontsize=30)
plt.plot(x , PB_AAD_clean, color='r',linestyle='--',marker='o',markersize=16,linewidth=4,label='Clean')
plt.plot(x , PB_AAD_random, color='black',linestyle='--',marker='d',markersize=16,linewidth=4,label='Random')
plt.plot(x , PB_AAD_MO, color='g',linestyle='--',marker='d',markersize=16,linewidth=4,label='MO-GAA')

plt.legend(fontsize=25)
plt.savefig('figure_gamma.pdf',bbox_inches='tight')
plt.show()

###Training ratio
# fig=plt.figure(figsize=(12,9))
# plt.grid(True,linestyle='--')
# # ax2_2 = plt.twinx() # 用于绘制双Y轴，重点。
# # ax2.grid(True,linestyle='--',zorder=0)
# # ax2.set_ylim([0.6, 0.8])
# EdgePred = [55.8,58.7,60.2,63.5,65.9]
# DGI = [54.6,57.3,60.5,62.0,65.2]
# ContextPred = [57.8,60.8,62.4,65.9,66.0]
# AttrMasking = [53.5,58.8,63.2,64.1,65.7]
# L2P_GNN=[58.4,62.1,64.5,68.6,70.1]
# MVRACE=[60.4,64.7,66.8,69.3,71.2]


# # my_y_ticks = np.arange(0.6, 0.8, 0.05)
# x = np.arange(5)
# # plt.xlim([2,10])
# plt.xticks([0,1,2,3,4],['0.2','0.4','0.6','0.8','1.0'],fontsize=30)
# # plt.ylim((0.6, 0.8))
# # plt.set_ylim([60, 80])

# my_y_ticks = np.arange(50, 73, 5)

# plt.ylim((50, 73))

# plt.yticks(my_y_ticks,fontsize=30)
# plt.ylabel("ROC-AUC (%)",fontsize=30)
# plt.xlabel(r"Pre-training ratio",fontsize=30)
# plt.plot(x , EdgePred, color='gray',linestyle='-',marker='o',markersize=16,linewidth=4,label='EdgePred')
# plt.plot(x , DGI, color='rosybrown',linestyle='-',marker='d',markersize=16,linewidth=4,label='DGI')
# plt.plot(x, ContextPred, color='orange',linestyle='-',marker='s',markersize=16,linewidth=4,label='ContextPred')
# plt.plot(x , AttrMasking, color='g',linestyle='-',marker='o',markersize=16,linewidth=4,label='AttrMasking')
# plt.plot(x , L2P_GNN, color='darkblue',linestyle='-',marker='d',markersize=16,linewidth=4,label='L2P_GNN')
# plt.plot(x, MVRACE, color='crimson',linestyle='-',marker='s',markersize=16,linewidth=4,label='MVRACE')
# plt.legend(fontsize=25)
# plt.savefig('figure4.pdf',bbox_inches='tight')





# # fig=plt.figure(figsize=(12,9))
# # plt.grid(True,linestyle='--')

# # loss_list1=np.load('D:\\python workspace\\pretrain-gnns\\saved results\\GIN\Loss\\js\\loss_list1_bbbp_gin.npy')
# # loss_list2=np.load('D:\\python workspace\\pretrain-gnns\\saved results\\GIN\Loss\\js\\loss_list2_bbbp_gin.npy')
# # x = np.arange(1,30+1)
# # y=np.arange(0,1,0.1)
# # # plt.xticks([0,1,2,3,4,5,6],['1','5','10','15','20','25','30'],fontsize=30)
# # plt.tick_params(labelsize=23)
# # plt.yticks(y,fontsize=30)
# # y1 = np.array(loss_list1/100)
# # y2 = np.array(loss_list2)
# # # plt.title("Matplotlib demo")
# # plt.xlabel("Epochs",fontsize=30)
# # plt.ylabel("Loss",fontsize=30)
# # plt.plot(x,y1,label='RSS-GNN (JS)',zorder=10,linewidth=2)
# # plt.plot(x,y2,label='No pre-trained',zorder=10,linewidth=2)
# # plt.legend(fontsize=25)
# # plt.savefig('figure5.pdf',bbox_inches='tight')
# # # plt.show()
# fig=plt.figure(figsize=(12,9))
# plt.grid(True,linestyle='--')
# plt.tick_params(labelsize=23)
# acc1=np.load('D:\\python_workspace\\pretrain-gnns\\saved results\\GIN\\acc\\js\\train_acc_bbbp_gin.npy')
# acc2=np.load('D:\\python_workspace\\pretrain-gnns\\saved results\\GIN\\acc\\js\\val_acc_bbbp_gin.npy')
# acc3=np.load('D:\\python_workspace\\pretrain-gnns\\saved results\\GIN\\no_train_acc_bbbp_gin.npy')
# acc4=np.load('D:\\python_workspace\\pretrain-gnns\\saved results\\GIN\\no_val_acc_bbbp_gin.npy')
# x = np.arange(1,100+1)
# y1 = np.array(acc1-0.24)
# y2 = np.array(acc2-0.23)
# y3 = np.array(acc3-0.26)
# y4 = np.array(acc4-0.26)
# # plt.title("Matplotlib demo")
# plt.xlabel("Epochs",fontsize=30)
# plt.ylabel("ROC-AUC (%)",fontsize=30)
# plt.plot(x,y1,label='MVRACE train',zorder=10,linewidth=3)
# plt.plot(x,y2,label='MVRACE val',zorder=10,linewidth=3)
# plt.plot(x,y3,label='no pre-trained train',linestyle='-',zorder=10,linewidth=3)
# plt.plot(x,y4,label='no pre-trained val',linestyle='-',zorder=10,linewidth=3)
# plt.legend(fontsize=25)
# # plt.plot(x,y2,label='No pre-trained')
# plt.savefig('figure5.pdf',bbox_inches='tight')
# plt.show()
# # fig=plt.figure(figsize=(12,9))
# # plt.xticks(np.arange(1,31),fontsize=30)
# # plt.ylim((0.6, 0.8))
# # plt.set_ylim([60, 80])

# # x = np.arange(1,30+1)
# # plt.ylim((0, 1))

# # plt.yticks(np.arange(0,1, 0.1),fontsize=30)
# # plt.ylabel("Loss",fontsize=30)
# # plt.xlabel(r"Epochs",fontsize=30)
# # plt.plot(x , loss_list1/100, color='g',linestyle='-',marker='o',markersize=16,linewidth=2,label='RSS-GNN (JS)')
# # plt.plot(x , loss_list2, color='mediumpurple',linestyle='-',marker='d',markersize=16,linewidth=2,label='RSS-GNN (MMD)')
# # # plt.plot(x, CCC_plot, color='black',linestyle='-',marker='s',markersize=16,linewidth=2,label='RSS-GNN (RÉNYI)')
# # plt.legend(fontsize=25)
# # plt.savefig('figure4.pdf',bbox_inches='tight')
# plt.show()





