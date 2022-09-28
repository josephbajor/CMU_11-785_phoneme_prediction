from dataclasses import dataclass
import os
import torch

@dataclass
class Hparams:
    ### Data ###
    datapath:os.PathLike = '/Users/josephbajor/Dev/Datasets/11-785-f22-hw1p2/dev-clean'
    architecture_name:str = 'placeholder'

    ### Model Params ###
    context:int = 25
    layers:int = 20
    width:int = 1024
    dropout_p:float = 0.2
    

    ### Train Params ###
    lr:float = 0.001
    batch_size:int = 1024






    def to_config(self):
        return self.asdict()
