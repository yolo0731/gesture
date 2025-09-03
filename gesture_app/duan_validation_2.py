# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 10:10:15 2022

@author: admin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
torch.__version__
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

BATCH_SIZE = 512 #大概需要2G的显存
EPOCHS = 2 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多
DOWNLOAD_MNIST = False 



train_set =  datasets.EMNIST('data', train=True, split='letters',download=True, 
                transform=transforms.Compose([
                     lambda img: transforms.functional.rotate(img, -90),#逆时针旋转90度
                     lambda img: transforms.functional.hflip(img),#水平翻折
                     transforms.ToTensor(),
                     transforms.Normalize((0.1723,), (0.3309,))#归一化就是要把图片3个通道中的数据整理到[-1, 1]区间
                ]))
# for image,label in train_set:
#     image = image.numpy().transpose(1,2,0) 
#     std = [0.5]
#     mean = [0.5]
#     image = image * std + mean
#train_set,val_set=torch.utils.data.random_split(train_set,[87360,37440])
#train_set,val_set=torch.utils.data.random_split(train_set,[99840,24960])
train_set,val_set=torch.utils.data.random_split(train_set,[112320,12480])


test_set = datasets.EMNIST('data', 
                    split='letters',
                    download=False,
                    train=False, 
                    
                    transform=transforms.Compose([
                        lambda img: transforms.functional.rotate(img, -90),
                        lambda img: transforms.functional.hflip(img),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1723,), (0.3309,))
                   ])
                   
)
#下载训练集
train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=BATCH_SIZE, shuffle=True)
#下载测试集
test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE, shuffle=True)
#定义卷积神经网络
class ConvNet2(nn.Module):
    def __init__(self):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28,也就是说一次训练512张图片）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 32, 3) # 输入通道数1，输出通道数32，核的大小3,输出通道数由核的数量决定
        self.conv2 = nn.Conv2d(32, 128, 3) # 输入通道数32，输出通道数128，核的大小3
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 512, 2)
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(512, 768) # 输入通道数是2000，输出通道数是500（输入由卷积层和池化层计算）
        self.fc2 = nn.Linear(768, 27) # 输入通道数是500，输出通道数是10，即10分类
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.25)
        self.b1 = nn.BatchNorm2d(32)
        self.b2 = nn.BatchNorm2d(128)
        self.b3 = nn.BatchNorm2d(256)
        self.b4 = nn.BatchNorm1d(768)
    def forward(self,x):
        in_size = x.size(0) # 在本例中in_size=512，也就是BATCH_SIZE的值。输入的x可以看成是512*1*28*28的张量。
        
        out = self.conv1(x) # batch*1*28*28 -> batch*32*26*26（28x28的图像经过一次核为5x5的卷积，输出变为24x24（28-5+1）
        out = self.b1(out)
        out = F.relu(out) # batch*32*26*26（激活函数ReLU不改变形状））
        out = F.max_pool2d(out, 2, 2) # batch*32*26*26 -> batch*32*13*13（2*2的池化层会减半）
        out = self.dropout2(out)
        
        out = self.conv2(out) # batch*32*13*13 -> batch*128*11*11（再卷积一次，核的大小是3）
        out = self.b2(out)
        out = F.relu(out) # batch*128*11*11
        out = F.max_pool2d(out, 2, 2, padding=1)
        out = self.dropout2(out)
        
        out = self.conv3(out)#4*4*256
        out = self.b3(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        out = self.dropout2(out)
        
        out = self.conv4(out)#512*1
        out = F.relu(out)
        
        out = out.view(in_size, -1) # batch*20*10*10 -> batch*2000（平铺，out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        
        out = self.fc1(out) # batch*512 -> batch*768
        out = self.b4(out)
        out = F.relu(out) # batch*500
        out = self.dropout1(out)
        
        out = self.fc2(out) # batch*768 -> batch*10
        out = F.log_softmax(out, dim=1) # 计算log(softmax(x))，分类后每个数字的概率值
        return out




#训练
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0: 
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss
def validation(model,device,val_loader):
     model.eval()
     valid_loss = 0
     val_correct = 0
     with torch.no_grad():
       
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                valid_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
                pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
                val_correct += pred.eq(target.view_as(pred)).sum().item()

        valid_loss /= len(val_loader.dataset)
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            valid_loss, val_correct, len(val_loader.dataset),
            100. * val_correct / len(val_loader.dataset)))
        return val_correct,valid_loss

#测试
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1] # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct,test_loss

if __name__ == '__main__':
    
    model = ConvNet().to(DEVICE)
    # model.summary()
    optimizer = optim.SGD(model.parameters(), lr=0.1,momentum=0.5)
    
    train_loss=[]
    
    val_losses=[]
    val_acc = []
    
    test_losses=[]
    acc=[]
    

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, DEVICE, train_loader, optimizer, epoch)
        val_correct,valid_loss = validation(model,DEVICE,val_loader)
        correct,test_loss = test(model, DEVICE, test_loader)
        train_loss.append(loss.item())
        
        val_acc.append(val_correct / len(val_loader.dataset))
        val_losses.append(valid_loss)
        acc.append(correct / len(test_loader.dataset))
        test_losses.append(test_loss)
        
        
    #可视化    
    x=np.arange(1,EPOCHS+1,1)
    
    plt.plot(x,val_losses,color= 'green',label = "Validation Loss")
    plt.plot(x,train_loss,color = 'blue',label = 'Train Loss')
    
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    plt.plot(x,test_losses,color = 'red',label = 'Test Loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    plt.plot(x,acc,color = 'red',label = 'Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()  
    
    #保存训练完成后的模型
    torch.save(model.state_dict(), './EMNIST2.pth')
    


