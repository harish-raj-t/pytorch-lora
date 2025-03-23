import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
torch.manual_seed(42)


class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(trainloader):
    epochs = 1  
    for epoch in range(epochs):
        for images, labels in trainloader:
            images = images.view(images.shape[0], -1).to(device) 
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print('Epoch:', epoch, 'Loss:', loss.item())
    

def test(testloader):
    with torch.no_grad():
        correct = 0
        total = 0
        wrong_counts = [0 for i in range(10)]
        for images, labels in testloader:
            images = images.view(images.shape[0], -1).to(device) 
            labels = labels.to(device)
            output = model(images)
            for idx, i in enumerate(output):
                if torch.argmax(i) == labels[idx]:
                    correct +=1
                else:
                    wrong_counts[labels[idx]] +=1
                total +=1
        print(f'Accuracy: {round(correct/total, 3)}')
        for i in range(len(wrong_counts)):
            print(f'wrong counts for the digit {i}: {wrong_counts[i]}')

train(trainloader)
test(testloader)

original_weights = {}
for name, param in model.named_parameters():
    original_weights[name] = param.clone().detach()

total_parameters_original = 0
for index, layer in enumerate([model.fc1, model.fc2, model.fc3]):
    total_parameters_original += layer.weight.nelement() + layer.bias.nelement()
    print(f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape}')
print(f'Total number of parameters: {total_parameters_original:,}')

class LoRA(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device = 'cpu'):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros((rank,features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        self.scale = alpha/rank
        self.enabled = True

    def forward(self, x):
        if self.enabled:
            x = x + torch.matmul(self.lora_B, self.lora_A).view(x.shape) * self.scale
        return x
    
import torch.nn.utils.parametrize as parametrize

def linear_layer_parameterization(layer, device, rank=2, lora_alpha=1):
    features_in, features_out = layer.weight.shape
    return LoRA(features_in, features_out, rank=rank, alpha=lora_alpha, device = device)

parametrize.register_parametrization(
    model.fc1, "weight", linear_layer_parameterization(model.fc1, device)
)
parametrize.register_parametrization(
    model.fc2, "weight", linear_layer_parameterization(model.fc2, device)
)
parametrize.register_parametrization(
    model.fc3, "weight", linear_layer_parameterization(model.fc3, device)
)

def enable_disable_lora(enabled=True):
    for layer in [model.fc1, model.fc2, model.fc3]:
        layer.parametrizations["weight"][0].enabled = enabled


total_parameters_lora = 0
total_parameters_non_lora = 0
for index, layer in enumerate([model.fc1, model.fc2, model.fc3]):
    total_parameters_lora += layer.parametrizations["weight"][0].lora_A.nelement() + layer.parametrizations["weight"][0].lora_B.nelement()
    total_parameters_non_lora += layer.weight.nelement() + layer.bias.nelement()
    print(
        f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape} + Lora_A: {layer.parametrizations["weight"][0].lora_A.shape} + Lora_B: {layer.parametrizations["weight"][0].lora_B.shape}'
    )
assert total_parameters_non_lora == total_parameters_original
print(f'Total number of parameters (original): {total_parameters_non_lora:,}')
print(f'Total number of parameters (original + LoRA): {total_parameters_lora + total_parameters_non_lora:,}')
print(f'Parameters introduced by LoRA: {total_parameters_lora:,}')
parameters_incremment = (total_parameters_lora / total_parameters_non_lora) * 100
print(f'Parameters incremment: {parameters_incremment:.3f}%')

for name, param in model.named_parameters():
    if 'lora' not in name:
        print(f'Freezing non-LoRA parameter {name}')
        param.requires_grad = False


enable_disable_lora()

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
exclude_indices = trainset.targets == 9
trainset.data = trainset.data[exclude_indices]
trainset.targets = trainset.targets[exclude_indices]

train_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

train(train_loader)
test(testloader)

assert torch.all(model.fc1.parametrizations.weight.original == original_weights['fc1.weight'])
assert torch.all(model.fc2.parametrizations.weight.original == original_weights['fc2.weight'])
assert torch.all(model.fc3.parametrizations.weight.original == original_weights['fc3.weight'])

enable_disable_lora(enabled=True)
assert torch.equal(model.fc1.weight, model.fc1.parametrizations.weight.original + (model.fc1.parametrizations.weight[0].lora_B @ model.fc1.parametrizations.weight[0].lora_A) * model.fc1.parametrizations.weight[0].scale)

enable_disable_lora(enabled=False)
assert torch.equal(model.fc1.weight, original_weights['fc1.weight'])