# Owen Kunhardt
# CSCI 350: 2048 Project
# Implementation for custom dataloader class for 2048 data

import torch
import numpy as np
import pandas as pd

# Class to load the 2048 dataset
class Dataset2048:
    
    def __init__(self, path, threshold = 0, threshold_index = -1, classification = False):
        
        df = pd.read_csv(path, header=None)
        
        # If threshold_index is specified
        # This is used to remove early game data based on a certain heuristic
        if threshold_index > -1:
            
            df = df.drop(df[df[threshold_index] < threshold].index)
            
        self.data_len = len(df)
        
        input_data = df.to_numpy()
        
        # Shuffles rows of data
        np.random.shuffle(input_data)
        
        # Extracts targets from data and takes the log_2 of them
        self.targets = np.log(input_data[:,-1].reshape((input_data.shape[0],1)))/np.log(2)
        
        # If task is classification, one-hot encodes target data
        if classification:
            
            max_val = np.amax(self.targets)
            min_val = np.amin(self.targets)
    
            self.num_classes = int(max_val - min_val + 1)
    
            self.targets = self.targets - min_val
            
            temp = [convert(int(i), self.num_classes) for i in self.targets]
            
            self.targets = np.concatenate(temp, axis = 0)
            
            # Converts data to Pytorch tensor
            self.targets = torch.from_numpy(self.targets).long()
        
        else:
            
            # Converts data to Pytorch tensor
            self.targets = torch.from_numpy(self.targets).float()
        
        # Removes targets from data and sets that as instances data
        self.instances = np.delete(input_data, input_data.shape[1]-1, axis=1)
        
        # Converts data to Pytorch tensor
        self.instances = torch.from_numpy(self.instances).float()
        
        
        
    # Returns instance and label 
    def __getitem__(self, index):
        
        instance = self.instances[index, :]
        target = self.targets[index, :]
        
        return (instance, target)
    
        
    def __len__(self):
        
        return self.data_len

# Used to create one-hot encoding for an instance   
def convert(pos, num_classes):
    
    target = np.zeros((num_classes, 1))
    target[pos] = 1
    return target.T   
    
    
    
    
    
    
    
