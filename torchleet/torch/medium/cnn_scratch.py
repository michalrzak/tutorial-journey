import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem: Implement a CNN for CIFAR-10 (With Custom Layers)

    ### Problem Statement
    You are tasked with implementing a **Convolutional Neural Network (CNN)** for image classification on the **CIFAR-10** dataset using PyTorch. However, instead of using PyTorch's built-in `nn.Conv2d` and `nn.MaxPool2d`, you must implement these layers **from scratch** using `nn.Module`. Your model will include convolutional layers for feature extraction, pooling layers for downsampling, and fully connected layers for classification.

    ### Requirements
    1. **Implement Custom Layers**:
       - Create a custom `Conv2dCustom` class that mimics the behavior of `nn.Conv2d`.
       - Create a custom `MaxPool2dCustom` class that mimics the behavior of `nn.MaxPool2d`.

    2. **Define the CNN Model**:
       - Use `Conv2dCustom` for convolutional layers.
       - Use `MaxPool2dCustom` for pooling layers.
       - Use standard `nn.Linear` for fully connected layers.
       - The model should process input images of shape `(3, 32, 32)` as in the CIFAR-10 dataset.

    ### Constraints
    - You must not use `nn.Conv2d` or `nn.MaxPool2d`. Use your own custom implementations.
    - The CNN should include multiple convolutional and pooling layers, followed by fully connected layers.
    - Ensure the model outputs class predictions for **10 classes**, as required by CIFAR-10.

    <details>
      <summary>💡 Hint</summary>
      Define `Conv2dCustom` and `MaxPool2dCustom` as subclasses of `nn.Module`. Use nested loops and tensor slicing to perform the operations.
      In `CNNModel.__init__`, use these custom layers to build the architecture.
      Implement the forward pass to pass inputs through convolution, activation, pooling, flattening, and fully connected layers.
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
    import torch.nn.functional as F
    return nn, optim, torch, torchvision, transforms


@app.cell
def _(torch, torchvision, transforms):
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return test_loader, train_dataset, train_loader


@app.cell
def _(Conv2dCustom, nn):
    # Define the CNN Model
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Output: 32x32x32
            # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64x32x32
            self.conv1 = Conv2dCustom(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = Conv2dCustom(32, 64, kernel_size=3, stride=1, padding=1)
        
            # self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 64x16x16
            self.pool = nn.MaxPool2dCustom(kernel_size=2, stride=2)  # Output: 64x16x16
            self.fc1 = nn.Linear(64 * 16 * 16, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)  # Flatten
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    return (CNNModel,)


@app.cell
def _(nn, stride, torch):
    class Conv2dCustom(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.kernels = nn.Parameter(torch.rand(
                (self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
            ))
            self.bias = nn.Parameter(torch.zeros(self.out_channels))

            self.stride = stride
            self.padding = padding

        def forward(self, x):
            h_out = int(
                (x.shape[-2] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride)
            w_out = int(
                (x.shape[-1] + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride)
            outp = torch.rand((x.shape[0], self.out_channels, h_out, w_out))

            x_padded = torch.zeros((
                x.shape[0],
                x.shape[1],
                self.padding + x.shape[2] + self.padding,
                self.padding + x.shape[3] + self.padding
            ))
            x_padded[
                :,
                :,
                self.padding:self.padding+x.shape[2],
                self.padding:self.padding+x.shape[3]] = x
        
            for i_batch in range(x.shape[0]):
                for i_out in range(self.out_channels):
                    kernel = self.kernels[i_out, :, :, :]
                    for i_x in range(h_out):
                        for i_y in range(w_out):
                            i_x_adj = i_x * self.stride
                            i_y_adj = i_y * self.stride
                            chunk = x_padded[
                                i_batch,
                                :,
                                i_x_adj:(i_x_adj + self.kernel_size),
                                i_y_adj:(i_y_adj + self.kernel_size)]

                            point = torch.sum(chunk * kernel) + self.bias[i_out]
                            outp[i_batch, i_out, i_x, i_y] = point
            return outp

    class MaxPool2dCustom(nn.Module):
        def __init__(self, kernel_size, stride=None):
            self.kernel_size = kernel_size
            self.stride = stride if stride is not None else kernel_size

        def forward(self, x):
            h_out = int((x.shape[-2] + (self.kernel_size - 1) - 1) / stride + 1)
            w_out = int((x.shape[-1] + (self.kernel_size - 1) - 1) / stride + 1)

            outp = torch.rand((x.shape[0], x.shape[1], h_out, w_out))
            for i_batch in range(x.shape[0]):
                for i_dim in range(x.shape[1]):
                    for i_h in range(h_out):
                        for i_w in range(w_out):
                            i_h_adj = i_h * self.stride
                            i_w_adj = i_w * self.stride
                            chunk = x[
                                i_batch,
                                i_dim,
                                i_h_adj:i_h_adj + self.kernel_size,
                                i_w_adj:i_w_adj + self.kernel_size]
                            outp[i_batch, i_dim, i_h, i_w] = torch.max(chunk)
            return outp
    return (Conv2dCustom,)


@app.cell
def _(CNNModel, nn, optim, train_loader):
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
def _(model, test_loader, torch):
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
def _(model):
    list(model.conv2.parameters())[0].shape
    return


@app.cell
def _(train_dataset):
    train_dataset[0][0].shape
    return


@app.cell
def _(Conv2dCustom, torch):
    conv2d = Conv2dCustom(3, 128, 3, padding=1, stride=2)
    test = torch.rand((1, 3, 30, 30))
    with torch.no_grad():
        x = conv2d(test)
    return (x,)


@app.cell
def _(x):
    x.shape
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
