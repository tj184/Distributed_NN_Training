
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from model_parts import ModelPart2
from send_utils import send_tensor, receive_tensor
from config import PC1_IP, PORT_FORWARD, PORT_BACKWARD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PC2")

def main():
    device = torch.device("cpu")
    model2 = ModelPart2().to(device)
    optimizer = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    logger.info("Waiting for input from PC1...")
    while True:
        try:
            intermediate, target = receive_tensor(PORT_FORWARD)
            intermediate.requires_grad = True
            target = target.to(device)

            optimizer.zero_grad()
            output = model2(intermediate)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            send_tensor(intermediate.grad, PC1_IP, PORT_BACKWARD)
            logger.info(f"Processed batch, Loss: {loss.item():.4f}")

        except Exception as e:
            logger.error(f"Processing failed: {e}")

if __name__ == "__main__":
    main()
