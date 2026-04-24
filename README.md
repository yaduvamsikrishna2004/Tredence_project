# Tredence_project
Self-Pruning Neural Network

Overview
This project implements a self-pruning neural network that learns to remove unnecessary connections during training. Unlike traditional pruning methods that are applied after training, this approach introduces learnable gate parameters that dynamically control the importance of each weight.

The model is trained on the CIFAR-10 dataset using PyTorch. A sparsity regularization term is added to the loss function to encourage the network to reduce the number of active connections.

---

PrunableLinear Layer
A custom linear layer is implemented instead of using the standard linear layer.

Each weight is associated with a learnable gate score. During the forward pass, the gate scores are passed through a sigmoid function to produce values between 0 and 1. These gate values are multiplied element-wise with the weights.

If a gate value approaches zero, the corresponding weight is effectively removed. If it is close to one, the weight remains active. This allows the network to learn which connections are important.

---

Neural Network Architecture
The network is a feed-forward model consisting of three layers:

Input layer: 3072 features (flattened 32x32x3 image)
Hidden layer 1: 1024 neurons
Hidden layer 2: 512 neurons
Output layer: 10 classes

All layers use the custom PrunableLinear module.

---

Loss Function
The total loss is a combination of classification loss and sparsity loss:

Total Loss = CrossEntropyLoss + lambda × SparsityLoss

The classification loss ensures correct predictions, while the sparsity loss encourages pruning.

The sparsity loss is computed as the sum of all gate values across the network. This corresponds to L1 regularization, which promotes sparsity.

---

Why L1 Encourages Sparsity
L1 regularization pushes many values toward zero. Since gate values are constrained between 0 and 1, minimizing their sum encourages many gates to become very small or zero.

When a gate becomes zero, the corresponding weight is effectively removed. This results in a sparse network where only important connections remain active.

---

Results

Lambda        Test Accuracy (%)        Sparsity Level (%)
1e-05         58.73                    0.42
0.0001        57.31                    2.81
0.001         55.46                    3.76

---

Analysis
As the value of lambda increases, the sparsity level increases slightly, while the accuracy decreases. However, the sparsity remains relatively low for the chosen lambda values, indicating that the regularization strength is not strong enough to enforce significant pruning.

This demonstrates the trade-off between model performance and sparsity. Stronger regularization leads to more pruning but may reduce accuracy.

---

Gate Value Distribution
A histogram of the gate values shows that most gate values are concentrated away from zero, indicating limited pruning. A stronger sparsity penalty would result in a larger number of gate values near zero.

---

Conclusion
The self-pruning neural network successfully integrates learnable gates to control weight importance. While the model demonstrates the pruning mechanism, the sparsity level depends heavily on the choice of the regularization parameter.

This approach is useful for reducing model size and improving efficiency in resource-constrained environments such as mobile and embedded systems.

---

How to Run

Install dependencies:
pip install torch torchvision matplotlib

Run the script:
python self_pruning_nn.py

It is recommended to use a GPU-enabled environment such as Google Colab for faster training.

---

Summary
The project demonstrates a neural network that can adapt its own structure during training by removing unnecessary connections. This reduces model complexity while maintaining reasonable performance.
