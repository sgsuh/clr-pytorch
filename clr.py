#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 16:04:43 2018

@author: seukgyo
"""

import numpy as np

class CLR(object):
    def __init__(self, optimizer, base_lr=0.001, max_lr=0.006, step_size=500.0,
                 mode='triangular', gamma=1.0, scale_fn=None, 
                 scale_mode='cycle', last_iteration=-1):
        super(CLR, self).__init__()
        
        self.optimizer = optimizer
        
        self.base_lr = base_lr
        self.max_lr = max_lr
        
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.0
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.0**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: self.gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
            
        self.step(last_iteration+1)
        self.last_iteration = last_iteration
       
    def get_lr(self):
        cycle = np.floor(1 + self.last_iteration / (2 * self.step_size))
        x = np.abs(self.last_iteration/self.step_size - 2*cycle + 1)
                
        base_height = (self.max_lr - self.base_lr) * np.maximum(0, (1-x))
            
        if self.scale_mode == 'cycle':
            lr = self.base_lr + base_height * self.scale_fn(cycle)
        else:
            lr = self.base_lr + base_height * self.scale_fn(self.last_iteration)
                
        return lr
        
    def step(self, batch_iteration=None):
        if batch_iteration is None:
            batch_iteration = self.last_iteration + 1
        
        self.last_iteration = batch_iteration
            
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            