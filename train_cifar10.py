#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:37:50 2018

@author: seukgyo
"""

import torch
import torchvision
import torchvision.transforms as transforms

# The output of torchvision datasets are PILImage images of range [0, 1]
# Transform them to Tensors of Normalized Range [-1, 1]

transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

# Training on GPU

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define a Convolution Neural Network

from caffe_cifar10 import CIFAR10_QUICK

net = CIFAR10_QUICK()
net = net.to(device)

# Define a Loss Function

import torch.nn as nn

criterion = nn.CrossEntropyLoss()

import torch.optim as optim
import clr

total_epoch = 50

optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = clr.CLR(optimizer)

# Train the network    
for epoch in range(total_epoch):
    net.train()
    
    running_loss = 0.0
    
    for data in trainloader:
        
        scheduler.step()
        
        # get the inputs
        inputs, labels = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # zero the parameters gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        #%%
        optimizer.step()
        
        running_loss += loss.item()
        
    print('epoch: %d, loss: %.3f' % (epoch+1, running_loss))
    
print('Finished Training')

# Test the network on the test data
net.eval()

total = 0
correct = 0

for data in testloader:
    images, labels = data
    
    images = images.to(device)
    labels = labels.to(device)
    
    outputs = net(images)
    
    _, pred = torch.max(outputs, 1)
    
    total += labels.size(0)
    correct += (pred == labels).sum().item()
    
accuracy = correct / total

print('Accuracy : %.4f' % (accuracy))