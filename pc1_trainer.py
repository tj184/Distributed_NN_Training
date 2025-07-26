
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torchvision import datasets, transforms
from model_parts import ModelPart1
from send_utils import send_tensor, receive_tensor
from config import PC2_IP, PORT_FORWARD, PORT_BACKWARD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PC1")

def main():
    device = torch.device("cpu")
    model1 = ModelPart1().to(device)
    optimizer = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=32, shuffle=True
    )

    for epoch in range(1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            intermediate = model1(data)

            try:
                
                send_tensor((intermediate.detach(), target), PC2_IP, PORT_FORWARD)

            
                grad = receive_tensor(PORT_BACKWARD)
                intermediate.backward(grad)
                optimizer.step()

                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx} completed")

            except Exception as e:
                logger.error(f"Communication failed: {e}")

if __name__ == "__main__":
    main()
