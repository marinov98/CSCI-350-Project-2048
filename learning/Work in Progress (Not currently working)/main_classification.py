# Owen Kunhardt
# CSCI 350: 2048 Project
# Main file to run classification neural network for the 2048 Dataset

import torch
import torch.nn as nn
import torch.utils.data.dataset
import dataloader
import nets

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Hyperparameters
input_size = 21
num_layers = 10
num_hidden_units = 15
output_size = 1
num_epochs = 10
learning_rate = 0.005
batch_size = 1000
threshold_index = -1
threshold = 41
regression = True


train_set = dataloader.Dataset2048(path="2048_training_data.csv", threshold_index = threshold_index, threshold = threshold, classification = True)
test_set = dataloader.Dataset2048(path="2048_testing_data.csv", threshold_index = threshold_index, threshold = threshold, classification = True)

print("Size of training set:", len(train_set))
print("Size of test set:", len(test_set))

# Loads training data into Pytorch dataloader to allow batch training
train_loader = torch.utils.data.DataLoader(dataset = train_set, batch_size=batch_size, shuffle=True)

model = nets.NeuralNet(input_size, num_hidden_units, num_layers, output_size).to(device)
    
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)

# Train the model
for epoch in range(num_epochs):

    for i, (instances, targets) in enumerate(train_loader):
    
        instances = instances.to(device)
        targets = targets.to(device)    
    
        # Forward pass
        outputs = model(instances)
        
        _, predicted = torch.max(outputs.data, 1)
    
        print(predicted.transpose(-1, 0))
        print(targets)
    
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 5000 == 0 or (i+1) == total_step:
            
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the model
with torch.no_grad():

    correct = 0
    total = 0
    
    instances = test_set.instances.to(device)
    labels = test_set.targets.to(device)

    outputs = model(instances)

    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

    print("Test set accuracy:", 100*correct/total)
    
# Save the model
torch.save(model.state_dict(), 'v9.pt')