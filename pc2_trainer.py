
import torch
import torch.nn as nn
import torch.optim as optim
from model_parts import ModelPart2
from utils import send_tensor, receive_tensor

HOST_PC1 = '127.0.0.1'     
PORT_FORWARD = 5001        
PORT_BACKWARD = 5002       

model2 = ModelPart2()
optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for step in range(10000):
    
    act1, target = receive_tensor(PORT_FORWARD)
    act1.requires_grad = True

    optimizer2.zero_grad()

    output = model2(act1)
    loss = criterion(output, target)
    loss.backward()

    
    send_tensor(act1.grad, HOST_PC1, PORT_BACKWARD)
    optimizer2.step()

    if step % 100 == 0:
        print(f"PC2: Step {step}, Loss: {loss.item():.4f}")
