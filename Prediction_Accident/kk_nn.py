import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch 
import functools
import time
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler

CUDA = torch.cuda.is_available()


def time_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        func_return_val = func(*args, **kwargs)
        end= time.perf_counter()
        print('{0:<1}.{1:<8} : {2:<8}sec'.format(func.__name__,'function',end-start))
        return func_return_val
    return wrapper


class focal_loss(nn.Module):    
    def __init__(self, alpha=0.25, gamma=2, num_classes = 3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)      
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算        
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数        
        :param labels:  实际类别. size:[B,N] or [B]        
        :return:
        """        
        # assert preds.dim()==2 and labels.dim()==1        
        preds = preds.view(-1,preds.size(-1))        
        self.alpha = self.alpha.to(preds.device)        
        preds_softmax = F.softmax(preds, dim=1) # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)        
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )        
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))        
        self.alpha = self.alpha.gather(0,labels.view(-1))        
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())        
        if self.size_average:        
            loss = loss.mean()        
        else:            
            loss = loss.sum()        
        return loss

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


    
    
class RoadDataset(Dataset):
    """Face Landmarks dataset."""
    
    def __init__(self, x,y):
        self.x = x
        self.y = y
    
    def get(self):
        sam = {int(i) :0 for i in set(self.y)}
        for y in self.y:
            sam[y] +=1
            
        return sam
                  
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()   
        x = torch.tensor(self.x[idx], dtype=torch.float)
        y = torch.tensor(self.y[idx], dtype=torch.long)         
        return  x, y
    
    
class NN(nn.Module):
    arch = [64]
    def __init__(self, in_dim, num_classes):
        super(NN, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, NN.arch[0]),
#             nn.ReLU(True),
            nn.BatchNorm1d(NN.arch[0]),
#             nn.Dropout(),
            
#             nn.Linear(NN.arch[0], NN.arch[1]),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.BatchNorm1d(NN.arch[1]),
            
#             nn.Linear(NN.arch[1], NN.arch[2]),
# #             nn.ReLU(True),
# #             nn.Dropout(),
#             nn.BatchNorm1d(NN.arch[2]),
            
            nn.Linear(NN.arch[-1], num_classes),
        )
        

    def forward(self, x):
        x = self.classifier(x)
        return  torch.sigmoid(x)
    
    
def train(loader, model, criterion, optimizer, epoch, warm=-1,warmup_scheduler=None):
    model.train()
    train_loss = 0
    for step, (x, y) in enumerate(loader):
        if CUDA:
            x = x.cuda()
            y = y.cuda()
#     for step in range(1 + len(data_x)//batch_size) :
#         if step !=len(data_x)//batch_size :
#             x = data_x[step*batch_size:(step+1)*batch_size]
#             y = data_y[step*batch_size:(step+1)*batch_size]
#         else:
#             x = data_x[step*batch_size:]
#             y = data_y[step*batch_size:]
        if CUDA:
            x = x.cuda()
            y = y.cuda()
        output = model(x) 
        
#         print(output.dtype, y.dtype)
        loss = criterion(output, y) 
        
        # Calculate gradients
        loss.backward()

        # Update parameters
        optimizer.step()
        
        if epoch <= warm and warmup_scheduler:
            warmup_scheduler.step()

#         print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
#             loss.item(),
#             optimizer.param_groups[0]['lr'],
#             epoch=epoch,
#             trained_samples=step *x.size()[0] + len(x),
#             total_samples=len(data_x)
#         ))
        train_loss += loss.item()
#     print('Training Epoch: {epoch}\t Loss:{:0.6f} '.format(
#         train_loss/(len(loader)*loader.batch_size), 
#         epoch=epoch,
#     ))

def evaluate(loader, model, criterion, epoch):
    model.eval()
    val_acc = 0
    valid_loss=0
    for step,(x,y) in enumerate(loader):
#     for step in range(1 + len(data_x)//batch_size) :
#         if step !=len(data_x)//batch_size :
#             x = data_x[step*batch_size:(step+1)*batch_size]
#             y = data_y[step*batch_size:(step+1)*batch_size]
#         else:
#             x = data_x[step*batch_size:]
#             y = data_y[step*batch_size:]
        if CUDA:
            x = x.cuda()
            y = y.cuda()
        
        pred = model(x) 
        
        loss = criterion(pred, y) 
        valid_loss += loss.item()
        _, pred = pred.topk(k=1, dim=1, largest=True, sorted=True)
#         print(pred, y)
        y = y.view(y.size(0), -1).expand_as(pred)
        correct = pred.eq(y).float()
        val_acc += correct[:, :1].sum()

    print('Evaluating Epoch: {epoch}\tLoss: {:0.4f}\t Accuracy: {:0.4f}%'.format(
        valid_loss/(len(loader)*loader.batch_size),
        100*val_acc/(len(loader)*loader.batch_size),
        epoch=epoch,
        
    ))
