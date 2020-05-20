# Owen Kunhardt
# CSCI 350: 2048 Project
# Implementation for the neural networks

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NeuralNet(nn.Module):
    
    def __init__(self, input_size, num_hidden_units, num_layers, output_size):
        
        super(NeuralNet, self).__init__()
        
        self.num_hidden_units = num_hidden_units
        self.num_layers = num_layers
        
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(input_size, num_hidden_units)
        
        self.layers = {}
        for l in range(2, num_layers):
            
            self.layers["fc" + str(l)] = nn.Linear(num_hidden_units, num_hidden_units)
            
        self.fc_final = nn.Linear(num_hidden_units, output_size)
        
    def forward(self, x):
        
        out = self.fc1(x)
        out = self.relu(out)
        
        for l in range(2, self.num_layers):
            
            layer = self.layers["fc" + str(l)].to(device)
            out = layer(out)
            out = self.relu(out)
        
        out = self.fc_final(out)
        
        return out
    
# Work in progress
class RecurrentNeuralNet(nn.Module):
    
    def __init__(self, input_size, num_hidden_units, num_layers, output_size):
        
        super(RecurrentNeuralNet, self).__init__()
        self.num_hidden_units = num_hidden_units
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, num_hidden_units, num_layers, batch_first=True)
        self.fc = nn.Linear(num_hidden_units, output_size)
        
    def forward(self, x):
        
        # Sets inital hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.num_hidden_units).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.num_hidden_units).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        
        return out

class Autoencoder(nn.Module):
    
    def __init__(self, input_size):
        
        self.encoder = nn.Sequential( 
            nn.Linear(input_size, 20), nn.ReLU(True), 
            nn.Linear(20, 15), nn.ReLU(True), 
            nn.Linear(15, 10), nn.ReLU(True), 
            nn.Linear(10, 5))

        self.decoder = nn.Sequential( 
            nn.Linear(5, 10), nn.ReLU(True), 
            nn.Linear(10, 15), nn.ReLU(True), 
            nn.Linear(15, 20), nn.ReLU(True), 
            nn.Linear(20, input_size))
        
    def forward(self, x):
        
        out = self.encoder(x)
        out = self.decoder(out)
        
        return out