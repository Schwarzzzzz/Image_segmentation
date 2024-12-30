import torch 
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset

class dataset(Dataset):
    
    def __init__(self):
        data = pd.read_csv('./SI_test.csv',names=['max','mean','root_mean_square','standard_deviation','y'])
        self.X = torch.FloatTensor(np.array(data.iloc[:,[0,1,2,3]]))
        self.y = torch.FloatTensor(np.array(data.iloc[:,[4]]))
        self.len = len(self.X)
        
    def __getitem__(self,index):
        
        return self.X[index],self.y[index]
    
    def __len__(self):
        
        return self.len
    
class Model(nn.Module):
    
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        y_pred = self.sigmoid(self.linear(x))
        return y_pred

def pred(data, weight_path):
    pred_list=[]
    model = Model()
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    for batch, X in enumerate(data):
        # 进行预测，也就是做了一次前向传播
        y_pred = model(X)
        y_pred = y_pred.data.item()
        if y_pred >= 0.5:
            yy_pred = 1
        else:
            yy_pred = 0
        pred_list.append(yy_pred)
    return pred_list



# 用验证集测试
if __name__ == '__main__':
    data = dataset()
    model = Model()
    model.load_state_dict(torch.load('./SIweight.pth'))
    model.eval()

    tp=0
    tn=0
    fp=0
    fn=0

    for batch, (X, y) in enumerate(data):

        # 进行预测，也就是做了一次前向传播
        y_pred = model(X)
        y_pred = y_pred.data.item()

        if y_pred>=0.5:
            yy_pred = 1
        else:
            yy_pred = 0
        print("%d\t"%batch,X,'\t',y,'\t',y_pred,'\t',yy_pred,end='')
        if yy_pred == y:
            print('\t正确')
            if yy_pred == 1:
                tp=tp+1
            else:
                tn=tn+1
        else:
            print('\t不正确')
            if yy_pred == 1:
                fp=fp+1
            else:
                fn=fn+1

    print('tp:'+str(tp))
    print('tn:'+str(tn))
    print('fp:'+str(fp))
    print('fn:'+str(fn))
