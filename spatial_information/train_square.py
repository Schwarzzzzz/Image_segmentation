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
        data = pd.read_csv('./SI_train_square.csv',names=['sepal_length','sepal_width','petal_length','petal_width','y'])
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
    

    
data = dataset()

model = Model()
# 使用BCE(Binary Cross Entropy)二元交叉熵损失函数
criterion = nn.BCELoss()
# 使用Adam优化算法
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 用于存放loss
loss_list = []

# 对整个样本训练次数
for epoch in range(10000):
    # 每次训练一个minibatch
    print(epoch+1)
    for i, (X, y) in enumerate(data):
        # 进行预测，也就是做了一次前向传播
        y_pred = model(X)
        # 计算损失
        loss = criterion(y_pred,y)
        # 梯度归0
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新梯度
        optimizer.step()
        # 记录损失
        loss_list.append(loss.data.item())

# 画出损失下降的图像
plt.plot(np.linspace(0,100,len(loss_list)),loss_list)
plt.show()
# 查看当前的训练参数，也就是w和b
print(model.state_dict())
torch.save(model.state_dict(), './SIweight_square.pth')
