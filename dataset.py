import torch
import os
import numpy as np
from hparams import Hparams

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, hparams:Hparams): # Feel free to add more arguments

        self.hparams = hparams

        self.mfcc_dir = f'{self.hparams.datapath}/mfcc' # TODO: MFCC directory - use partition to acces train/dev directories from kaggle data
        self.transcript_dir = f'{self.hparams.datapath}/transcript' # TODO: Transcripts directory - use partition to acces train/dev directories from kaggle data

        mfcc_names = sorted(os.listdir(self.mfcc_dir)) # TODO: List files in X_dir using os.listdir in sorted order, optionally subset using limit to slice the number of files you load
        transcript_names = sorted(os.listdir(self.transcript_dir)) # TODO: List files in Y_dir using os.listdir in sorted order, optionally subset using limit to slice the number of files you load

        assert len(mfcc_names) == len(transcript_names) # Making sure that we have the same no. of mfcc and transcripts

        self.mfccs, self.transcripts = [], []

        # TODO:
        # Iterate through mfccs and transcripts
        for i in range(0, len(mfcc_names)):
        #   Load a single mfcc
            mfcc = np.load(f'{self.hparams.datapath}/mfcc/{mfcc_names[i]}')
            
            mfcc = (mfcc.T - np.mean(mfcc, axis=1)).T
        #   Load the corresponding transcript and remove [SOS] and [EOS]
            transcript = np.load(f'{self.hparams.datapath}/transcript/{transcript_names[i]}')[1:-1]
        #   Append each mfcc to self.mfcc, transcript to self.transcript
            self.mfccs.append(mfcc)
            self.transcripts.append(transcript)

        

        # NOTE:
        # Each mfcc is of shape T1 x 15, T2 x 15, ...
        # Each transcript is of shape (T1+2) x 15, (T2+2) x 15 before removing [SOS] and [EOS]

        # TODO: Concatenate all mfccs in self.X such that the final shape is T x 15 (Where T = T1 + T2 + ...) 
        self.mfccs = np.concatenate(self.mfccs)

        # TODO: Concatenate all transcripts in self.Y such that the final shape is (T,) meaning, each time step has one phoneme output
        self.transcripts = np.concatenate(self.transcripts)
        # Hint: Use numpy to concatenate

        # Take some time to think about what we have done. self.mfcc is an array of the format (Frames x Features). Our goal is to recognize phonemes of each frame
        # From hw0, you will be knowing what context is. We can introduce context by padding zeros on top and bottom of self.mfcc
        self.mfccs = np.pad(self.mfccs, pad_width=((self.context,self.context),(0,0)), mode='constant')

        # These are the available phonemes in the transcript
        self.phonemes = [
            'SIL',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',  
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '<sos>', '<eos>']
        # But the neural network cannot predict strings as such. Instead we map these phonemes to integers
        
        self.phone_map = {phone:idx for idx, phone in enumerate(self.phonemes)}
        for idx, i in enumerate(self.transcripts):
            self.transcripts[idx] = self.phone_map[i]
        self.transcripts = self.transcripts.astype(int)

        # TODO: Map the phonemes to their corresponding list indexes in self.phonemes
        # Now, if an element in self.transcript is 0, it means that it is 'SIL' (as per the above example)

        # Length of the dataset is now the length of concatenated mfccs/transcripts
        #self.length = len(self.mfccs)
        #NO. BAD! Do not set the len to mfccs after padding, idiot

        self.length = len(self.transcripts)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        
        #print(f'context: {self.context}\nind: {ind}\nmfcc_ind: {ind+(self.context*2)+1}')
        frames = self.mfccs[ind:ind+(self.context*2)+1]
        frames = frames.flatten()

        frames = torch.FloatTensor(frames) # Convert to tensors
        phoneme = torch.tensor(self.transcripts[ind])

        return frames, phoneme


class AudioTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path:os.PathLike, context:int, offset=0, limit=None): # Feel free to add more arguments

        self.context = context
        self.offset = offset
        self.data_path = data_path

        self.mfcc_dir = f'{self.data_path}/mfcc' # TODO: MFCC directory - use partition to acces train/dev directories from kaggle data
        mfcc_names = os.listdir(self.mfcc_dir) # TODO: List files in X_dir using os.listdir in sorted order, optionally subset using limit to slice the number of files you load

        self.mfccs = []

        # TODO:
        # Iterate through mfccs and transcripts
        for i in range(0, len(mfcc_names)):
            mfcc = np.load(f'{self.data_path}/mfcc/{mfcc_names[i]}')
            self.mfccs.append(mfcc)

        self.mfccs = np.concatenate(self.mfccs)


        # Take some time to think about what we have done. self.mfcc is an array of the format (Frames x Features). Our goal is to recognize phonemes of each frame
        # From hw0, you will be knowing what context is. We can introduce context by padding zeros on top and bottom of self.mfcc
        self.mfccs = np.pad(self.mfccs, (self.context,self.context), 'constant')

        # Length of the dataset is now the length of concatenated mfccs/transcripts
        self.length = len(self.mfccs)

    def __len__(self):
        return self.length

    def __getitem__(self, ind):
        
        frames = self.mfccs[ind:ind+(self.context*2)+1]
        frames = frames.flatten()

        frames = torch.FloatTensor(frames) # Convert to tensors    

        return frames

hparams = Hparams()
test = AudioDataset(hparams)

next(iter(test))