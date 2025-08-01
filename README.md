Distributed Neural Network Training
A Python-based framework for distributed neural network training across multiple machines on the same local network. This project splits model layers between systems, enabling parallelized forward and backward passes for faster and more efficient training—ideal for setups with limited GPU/CPU resources per node.

🚀 Key Features
🔄 Layer-wise distribution of model computation

📶 Network-based communication for weight transfer between systems

⏱️ Efficient forward and backpropagation sync across devices

🔧 Modular design for easy extension to any PyTorch model

📦 Supports multiple training configurations and datasets

🧠 Concept
The model is split into two (or more) parts:

Part 1 is processed on Machine A: performs initial layers and sends activations

Part 2 is processed on Machine B: continues forward pass and computes loss

Backpropagation gradients are sent back in reverse to update both halves

This mimics pipeline parallelism and is ideal for setups with limited individual resources.

Usage

On Machine A (Client - Part 1):
python client.py --ip <server_ip> --port 8000

On Machine B (Server - Part 2):
python server.py --port 8000
