'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-21 14:26:54
LastEditors: ZhangHongYu
LastEditTime: 2021-01-23 22:28:00
'''
'''
Descripttion: 
Version: 1.0
Author: ZhangHongYu
Date: 2021-01-11 20:06:48
LastEditors: ZhangHongYu
LastEditTime: 2021-01-20 22:18:25
'''
import torch
import torchvision
import torchvision.transforms as transforms
import time
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import sys
from torch.nn import init
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hiddens = num_hiddens
        self.linear1 = nn.Linear(num_inputs, num_hiddens)
        self.linear2 = nn.Linear(num_hiddens, num_outputs)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1) #pytorch内置的loss已经包括softmax
        init.normal_(self.linear1.weight, mean=0, std=0.01)
        init.constant_(self.linear1.bias, val=0) 
        init.normal_(self.linear2.weight, mean=0, std=0.01)
        init.constant_(self.linear2.bias, val=0) 
    #此时X行n为样本个数，列m为输出的预测标签个数
    def forward(self, X):
        H = self.relu(self.linear1(X.view(-1,self.num_inputs)))
        O = self.linear2(H)
        return O

def accuracy(y_hat, y):
    return (y_hat.argmax(dim = 1) == y).float().mean().item()

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum +=  (net(X).argmax(dim = 1) == y).float().sum().item() #加起来最后一起除n
        n += y.shape[0]
    return acc_sum/n

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [ text_labels[int(i)] for i in labels]  #返回5个样本的预测label

        
def train(net, train_iter, test_iter, loss, num_epochs, batch_size, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif net.parameters() is not None:
                for param in net.parameters():
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                torch.optim.SGD(net.parameters(), lr)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train_acc %.3f, test_acc %.3f' % 
        (epoch+1, train_l_sum/n, train_acc_sum/n, test_acc))


def show_fashion_mnist(images, labels):
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

if __name__ =='__main__':
    mnist_train = torchvision.datasets.FashionMNIST(root='/mnt/mydisk/data'', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root='/mnt/mydisk/data'', train=False, download=True, transform=transforms.ToTensor())

    # print(type(mnist_train))
    # print(len(mnist_train),len(mnist_test))
    # feature, label = mnist_train[0]
    # print(feature.shape, label)

    batch_size = 256

    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4  # 多线程读取数据

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size, 
    shuffle = True,num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size = batch_size, 
    shuffle = True,num_workers=num_workers)

    # start = time.time()
    # for X, y in train_iter:
    #     continue
    # print("%.2f sec"% (time.time()-start))

    #输入特征和输出特征的维度，和样本个数无关
    num_inputs = 784
    num_outputs = 256
    num_hiddens = 100
    net = LinearNet(num_inputs, num_outputs, num_hiddens)
    
    # X = torch.rand((2,5))
    # X_prob = softmax(X)
    # print(X_prob, X_prob.sum(dim=1, keepdim=True))
    num_epochs = 5
    lr = 0.5

    optimizer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()
    train(net, train_iter, test_iter, loss, num_epochs, batch_size, lr, optimizer)

    torch.save(net.state_dict(), '/mnt/mydisk/model/model2.pt')
    m_state_dict = torch.load('/mnt/mydisk/model/model2.pt')
    net.load_state_dict(m_state_dict)

    X, y = iter(test_iter).next() #取一个batch，即5个样本
    true_labels = get_fashion_mnist_labels(y.numpy())
    pred_labels = get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
    titles = [true + '\n' + pred for true , pred in zip(true_labels, pred_labels)]
    
    show_fashion_mnist(X[0:9], titles[0:9])
    
