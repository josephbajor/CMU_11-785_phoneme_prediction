from dataclasses import dataclass
import os

@dataclass
class Hparams:

    ### Model Params ###
    context:int = 30
    layers:int = 8
    width:int = 1745
    dropout_p:float = 0.25
    
    ### Train Params ###
    lr:float = 1e-3
    batch_size:int = 1028
    epochs:int = 25
    mixed_precision:bool = False

    ### Data ###
    datapath:os.PathLike = '/Users/josephbajor/Dev/Datasets/11-785-f22-hw1p2/dev-clean'
    architecture:str = f'cyl_v12_H{layers}_W{width}_C{context}_noMP'


    def to_config(self):
        return self.asdict()