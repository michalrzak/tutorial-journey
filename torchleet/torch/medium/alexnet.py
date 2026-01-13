import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem: Write AlexNet from Scratch

    ### Problem Statement
    Implement the **AlexNet architecture** in PyTorch by completing the required sections. The model should include convolutional layers, ReLU activations, pooling layers, and fully connected layers to process image data for classification tasks.

    ### Requirements

    1. **Define the AlexNet Architecture**:
       - **Feature Extractor (Convolutional Base)**:
         - Stack convolutional layers with appropriate kernel sizes, strides, and paddings.
         - Use `nn.ReLU` as the activation function after each convolution.
         - Apply `nn.MaxPool2d` after selected layers to reduce spatial dimensions.
       - **Classifier (Fully Connected Layers)**:
         - Flatten the output from the convolutional base.
         - Add fully connected layers with ReLU activations and dropout for regularization.
         - End with a final linear layer projecting to the number of output classes.

    2. **Implement the Forward Method**:
       - Pass the input image through the convolutional base.
       - Flatten the feature map output to a vector.
       - Pass it through the fully connected classifier to produce final predictions.

    3. **Weight Initialization**:
       - Initialize weights of convolutional and linear layers using a normal distribution.
       - Set biases to zero.

    ### Constraints
    - Assume input images are RGB with shape **(3, 224, 224)**.
    - Ensure the model is compatible with **batch processing**.
    - The final layer output size should match the number of target classes (e.g., 10 for CIFAR-10).
    - Avoid using any pretrained models or high-level wrappers like `torchvision.models`.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    return DataLoader, nn, optim, torch, torchvision, transforms


@app.cell
def _(DataLoader, torchvision, transforms):
    # Load data
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize to AlexNet input
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    return test_loader, train_loader


@app.cell
def _(nn, torch):
    # Define AlexNet
    class AlexNet(nn.Module):
        def __init__(self, num_classes=10):  # Adjusted for CIFAR-10
            super().__init__()
            self.num_classes = num_classes
        
            self.conv1 = nn.Conv2d(3, 96, 11, stride=4)  # 54x54
            self.max_pool1 = nn.MaxPool2d(3, stride=2) # 26x26
            self.conv2 = nn.Conv2d(96, 256, 5, padding=2) # 26x26
            self.max_pool2 = nn.MaxPool2d(3, stride=2)
            self.conv3 = nn.Conv2d(256, 384, 3, padding=1)
            self.conv4 = nn.Conv2d(384, 384, 3, padding=1)
            self.conv5 = nn.Conv2d(384, 256, 3, padding=1)
            self.max_pool3 = nn.MaxPool2d(3, stride=2)
            self.activation1 = nn.ReLU()

            self.fc1 = nn.Linear(6400, 4096)
            self.fc2 = nn.Linear(4096, 4096)
            self.fc3 = nn.Linear(4096, self.num_classes)
            self.dropout = nn.Dropout(p=0.5)
            self.activation2 = nn.ReLU()


        def forward(self, x):
            x = self.conv1(x)  # 96x54x54
            x = self.max_pool1(x)  # 96x26x26
            x = self.activation1(x)

            x = self.conv2(x)  # 256x26x26
            x = self.max_pool2(x)  # 256x12x12
            x = self.activation1(x)

            x = self.conv3(x)  # 384x12x12
            x = self.activation1(x)

            x = self.conv4(x)  # 384x12x12
            x = self.activation1(x)

            x = self.conv5(x)  # 256x12x12
            x = self.max_pool3(x)  # 256x5x5
            x = self.activation1(x)

            x = torch.flatten(x, start_dim=1)  # 6400
        
            x = self.fc1(x)  # 4096
            x = self.activation2(x)
            x = self.dropout(x)

            x = self.fc2(x)  # 4096
            x = self.activation2(x)
            x = self.dropout(x)

            x = self.fc3(x)

            return x
        
    return (AlexNet,)


@app.cell
def _(AlexNet, nn, optim, torch):
    # --- Training setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    return criterion, device, model, optimizer


@app.cell
def _(criterion, device, model, optimizer, train_loader):
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        _correct = 0
        _total = 0
        for _images, _labels in train_loader:
            _images, _labels = (_images.to(device), _labels.to(device))
            _outputs = model(_images)
            loss = criterion(_outputs, _labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, _predicted = _outputs.max(1)
            _total += _labels.size(0)
            _correct += _predicted.eq(_labels).sum().item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, Accuracy: {100 * _correct / _total:.2f}%')
    return


@app.cell
def _(device, model, test_loader, torch):
    model.eval()
    _correct = 0
    _total = 0
    with torch.no_grad():
        for _images, _labels in test_loader:
            _images, _labels = (_images.to(device), _labels.to(device))
            _outputs = model(_images)
            _, _predicted = torch.max(_outputs.data, 1)
            _total += _labels.size(0)
            _correct += (_predicted == _labels).sum().item()
    print(f'Test Accuracy: {100 * _correct / _total:.2f}%')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
