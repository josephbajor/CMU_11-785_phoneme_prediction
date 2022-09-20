from dataset import AudioDataset
import torch


config = {
    'epochs': 10,
    'batch_size' : 1024,
    'context' : 1,
    'learning_rate' : 0.001,
    'architecture' : 'very-low-cutoff'
    # Add more as you need them - e.g dropout values, weight decay, scheduler parameters
}

if __name__ == '__main__':
    train_data = AudioDataset('/Users/josephbajor/Dev/Datasets/11-785-f22-hw1p2/train-clean-100', context=config['context']) #TODO: Create a dataset object using the AudioDataset class for the training data 

    train_loader = torch.utils.data.DataLoader(train_data, num_workers= 1,
                                            batch_size=config['batch_size'], pin_memory= True,
                                            shuffle= True)

    # Testing code to check if your data loaders are working
    for i, data in enumerate(train_loader):
        frames, phoneme = data
        print(frames.shape, phoneme.shape)
        break