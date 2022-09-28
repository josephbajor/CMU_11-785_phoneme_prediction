import torch
from torch import nn
import numpy as np

from hparams import Hparams

class DLHW1_MLP_modular(nn.Module):

    def __init__(self, hparams:Hparams, act_func):
        super.__init__()
        self.hparams = hparams

        imp_size = (2*self.hparams.context+1)
        out_size = 40

        self.layers = nn.ModuleList()


        self.layers.extend([
            nn.Linear(self.inp_size, self.hparams.width),
            nn.BatchNorm1d(self.hparams.width)
        ])

        for layer in range(len(self.hparams.layers)):
            if layer == 0:
                s1 = self.imp_size
                s2 = self.hparams.width

                self.layers.extend([
                    nn.Linear(s1, s2),
                    act_func(),
                    nn.Dropout(hparams.dropout_p),
                ])
                
            if layer == len(self.hparams.layers)-1:
                s1 = self.hparams.width
                s2 = self.out_size

                self.layers.extend([
                    nn.BatchNorm1d(s1),
                    nn.Linear(s1, s2),
                    act_func(),
                    nn.Softmax(s2)
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

        
        
