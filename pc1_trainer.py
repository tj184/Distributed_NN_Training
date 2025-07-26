
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model_parts import ModelPart1
from utils import send_tensor, receive_tensor
from config import PC1_IP, PC2_IP, PORT_FORWARD, PORT_BACKWARD

HOST = 'IP of PC2'   
PORT_FORWARD = 5001  
PORT_BACKWARD = 5002 

model1 = ModelPart1()
optimizer1 = optim.SGD(model1.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=32, shuffle=True
)

for epoch in range(1):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer1.zero_grad()

       
        act1 = model1(data)

        
        send_tensor((act1.detach(), target), HOST, PORT_FORWARD)

        grad_act1 = receive_tensor(PORT_BACKWARD)
        act1.backward(grad_act1)

        optimizer1.step()

        if batch_idx % 100 == 0:
            print(f"PC1: Batch {batch_idx} completed.")
