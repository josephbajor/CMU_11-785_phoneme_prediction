import torch
from torch import nn
import numpy as np

from hparams import Hparams

class DLHW1_MLP_modular(nn.Module):

    def __init__(self, hparams:Hparams, act_func):
        super().__init__()
        self.hparams = hparams

        self.imp_size = (2*self.hparams.context+1) * 15
        self.out_size = 40

        self.layers = nn.ModuleList()

        for layer in range(self.hparams.layers):
            if layer == 0:
                s1 = self.imp_size
                s2 = self.hparams.width

                self.layers.extend([
                    nn.Linear(s1, s2),
                    act_func(),
                    nn.Dropout(hparams.dropout_p),
                ])
                
            elif layer == self.hparams.layers-1:
                s1 = self.hparams.width
                s2 = self.out_size

                self.layers.extend([
                    nn.BatchNorm1d(s1),
                    nn.Linear(s1, s2),
                    nn.LogSoftmax(dim=1)
                ])

            else:
                s1 = self.hparams.width
                s2 = self.hparams.width

                self.layers.extend([
                    nn.BatchNorm1d(s1),
                    nn.Linear(s1, s2),
                    act_func(),
                    nn.Dropout(hparams.dropout_p),
                ])

            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight.data,nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight.data, 1)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def print_layers(self):
        print(self.layers)


        
        
