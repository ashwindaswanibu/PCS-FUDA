import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
def create_label_index(dataset_size, percent_labeled=0.05):
    return np.random.choice(dataset_size, int(dataset_size * percent_labeled), replace=False)
    
def get_fewshot_data_labels(dataset, indices):
    
    dataset = torch.utils.data.Subset(dataset, indices)
    return dataset

def get_unlabeled_data(dataset, indices):
    inverse_indices = np.setdiff1d(np.arange(len(dataset)), indices)
    dataset = torch.utils.data.Subset(dataset, inverse_indices)
    return dataset

def get_fewshot_dataloader(dataset, indices, batch_size=128):
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(indices))
    return train_loader

def get_unlabeled_dataloader(dataset, indices, batch_size=128):
    inverse_indices = np.setdiff1d(np.arange(len(dataset)), indices)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(inverse_indices))
    return train_loader





            
            