import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Problem: Implement a CNN for CIFAR-10 in PyTorch

    ### Problem Statement
    You are tasked with implementing a **Convolutional Neural Network (CNN)** for image classification on the **CIFAR-10** dataset using PyTorch. The model should contain convolutional layers for feature extraction, pooling layers for downsampling, and fully connected layers for classification. Your goal is to complete the CNN model by defining the necessary layers and implementing the forward pass.

    ### Requirements
    1. **Define the CNN Model**:
       - Add **convolutional layers** for feature extraction.
       - Add **pooling layers** to reduce the spatial dimensions.
       - Add **fully connected layers** to output class predictions.
       - The model should be capable of processing input images of size `(32x32x3)` as in the CIFAR-10 dataset.

    ### Constraints
    - The CNN should be designed with multiple convolutional and pooling layers followed by fully connected layers.
    - Ensure the model is compatible with the CIFAR-10 dataset, which contains 10 classes.


    <details>
      <summary>💡 Hint</summary>
      Add the convolutional (conv1, conv2), pooling (pool), and fully connected layers (fc1, fc2) in CNNModel.__init__.
      <br>
      Implement the forward pass to process inputs through these layers.
    </details>
    """)
    return


@app.cell
def _():
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader, train_loader


@app.class_definition
# Define the CNN Model
# TODO: Add convolutional, pooling, and fully connected layers
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3) # (16, 30, 30)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 32, 3) # (32, 28, 28)
        self.pool2 = nn.MaxPool2d(2, stride=2) # (32, 14, 14)
        self.activation2 = nn.ReLU()

        self.fully_connected1 = nn.Linear(32 * 14 * 14, 128)
        self.activation_fc1 = nn.ReLU()

        self.fully_connected2 = nn.Linear(128, 10)
        self.activation_fc2 = nn.Softmax()


    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.pool2(x)
        x = self.activation2(x)

        x = torch.reshape(x, (x.shape[0], 32 * 14 * 14))
        x = self.fully_connected1(x)
        x = self.activation_fc1(x)

        x = self.fully_connected2(x)
        x = self.activation_fc2(x)

        return x


@app.cell
def _(train_loader):
    # Initialize the model, loss function, and optimizer
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    # Training loop
    for epoch in range(epochs):
        for _images, _labels in train_loader:
            _outputs = model(_images)
            loss = criterion(_outputs, _labels)  # Forward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')  # Backward pass and optimization
    return (model,)


@app.cell
def _(model, test_loader):
    # Evaluate on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for _images, _labels in test_loader:
            _outputs = model(_images)
            _, predicted = torch.max(_outputs, 1)
            total += _labels.size(0)
            correct += (predicted == _labels).sum().item()
    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
