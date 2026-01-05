import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem: Add a Benchmark to Your PyTorch Code

    ### Problem Statement
    You are tasked with implementing a simple neural network model with fully connected layers and adding benchmarking functionality to measure and display the time taken for each epoch of training and testing. The goal is to evaluate the model's performance and record the time taken for both training and testing phases.

    ### Requirements
    1. **Define a Neural Network Model**:
       - Implement a simple feedforward neural network using fully connected layers (`nn.Linear`).
       - The network should be suitable for classification tasks.

    2. **Benchmark Training and Testing**:
       - Measure the time taken for each epoch during training and display the elapsed time.
       - Measure and display the time taken for the testing phase after each epoch.

    ### Constraints
    - The model should have at least two hidden layers with ReLU activations.
    - Use the appropriate loss function and optimizer for training the model.
    - Ensure that the benchmarking measures both the training and testing time accurately.

    <details>
      <summary>💡 Hint</summary>
      Define the SimpleNN class:
      <br>
      Add two fully connected layers:
      <br>
      Apply a ReLU activation function to the first layer.
      <br>
      <br>
      Benchmark the Code:
      <br>
      Measure and print training time for each epoch.
      <br>
      Measure and print testing time along with accuracy.
    </details>
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import time
    return nn, optim, time, torch, torchvision, transforms


@app.cell
def _(torch, torchvision, transforms):
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader, train_dataset, train_loader


@app.cell
def _(train_dataset):
    set(map(lambda ele: ele[1], train_dataset))
    return


@app.cell
def _(nn, optim, torch):
    # Define a simple neural network model
    # TODO: Add layers to the model
    class SimpleNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(1*28*28, 2000)
            self.activation1 = nn.ReLU()
        
            self.layer2 = nn.Linear(2000, 2000)
            self.activation2 = nn.ReLU()
        
            self.layer3 = nn.Linear(2000, 2000)
            self.activation3 = nn.ReLU()

            self.layer4 = nn.Linear(2000, 10)

        def forward(self, x):
            x = torch.flatten(x, start_dim=1)
            x = self.layer1(x)
            x = self.activation1(x)
            x = self.layer2(x)
            x = self.activation2(x)
            x = self.layer3(x)
            x = self.activation3(x)
            x = self.layer4(x)

            return x

    # Initialize the model, loss function, and optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    return criterion, model, optimizer


@app.cell
def _(criterion, model, optimizer, time, train_loader):
    # Training loop with benchmarking
    epochs = 5
    for epoch in range(epochs):
        _start_time = time.time()  # Start time for training
        for _images, _labels in train_loader:
            _outputs = model(_images)  # Forward pass
            loss = criterion(_outputs, _labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Backward pass and optimization
        _end_time = time.time()
        training_time = _end_time - _start_time
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Time: {training_time:.4f}s')  # End time for training
    return


@app.cell
def _(model, test_loader, time, torch):
    # Evaluate the model on the test set and benchmark the accuracy
    correct = 0
    total = 0
    _start_time = time.time()  # Start time for testing
    with torch.no_grad():
        for _images, _labels in test_loader:
            _outputs = model(_images)
            _, predicted = torch.max(_outputs, 1)
            total += _labels.size(0)
            correct += (predicted == _labels).sum().item()
    _end_time = time.time()
    testing_time = _end_time - _start_time  # End time for testing
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%, Testing Time: {testing_time:.4f}s')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
