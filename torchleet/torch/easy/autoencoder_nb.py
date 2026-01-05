import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem: Train an Autoencoder for Anomaly Detection

    ### Problem Statement
    You are tasked with implementing an **autoencoder** model for anomaly detection. The model will be trained on the **MNIST dataset**, and anomalies will be detected based on the reconstruction error. The autoencoder consists of an encoder to compress the input and a decoder to reconstruct the image. The difference between the original image and the reconstructed image will be used to detect anomalies.

    ### Requirements
    1. **Define the Autoencoder Architecture**:
        - **Encoder**:
            - Implement a series of convolutional layers followed by max-pooling layers.
            - The encoder should progressively reduce the spatial dimensions of the input image, capturing the most important features.
        - **Decoder**:
            - Implement a series of transposed convolutional layers (also known as deconvolutional layers) to upsample the compressed representation back to the original image size.
            - Use a **Sigmoid activation** function in the final layer to ensure that the output pixel values are between 0 and 1.

    2. **Forward Pass**:
       - Implement the forward method where the input image is passed through the encoder to obtain a compressed representation, followed by passing it through the decoder to reconstruct the image.

    ### Constraints
    - The autoencoder should work on the MNIST dataset, which consists of 28x28 grayscale images.
    - Ensure that the output of the decoder matches the original image size.
    - Use **Sigmoid activation** in the final layer to constrain the output pixel values between 0 and 1.

    <details>
      <summary>💡 Hint</summary>
      Focus on the encoder to downsample the input and the decoder to upsample and reconstruct the image.
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
    import numpy as np
    import matplotlib.pyplot as plt
    return nn, optim, plt, torch, torchvision, transforms


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
    train_dataset[0][0]
    return


@app.cell
def _(nn, optim):
    # Define an Autoencoder model
    # TODO: Implement the autoencoder architecture
    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 10, (3, 3)),  # 10 x 26 x 26
                nn.MaxPool2d((2,2)),  # 10 x 13 x 13
                nn.ReLU(),
                nn.Conv2d(10, 1, (3, 3)),  # 1 x 11 x 11
                nn.MaxPool2d((2, 2), stride=1),  # 1 x 10 x 10
                nn.ReLU(),
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(1, 5, (3, 3)),  # 5 x 12 x 12
                nn.ReLU(),
                nn.ConvTranspose2d(5, 10, (5, 5)),  # 10 x 16 x 16
                nn.ReLU(),
                nn.ConvTranspose2d(10, 10, (5, 5)),  # 10 x 20 x 20
                nn.ReLU(),
                nn.ConvTranspose2d(10, 10, (5, 5)),  # 10 x 24 x 24
                nn.ReLU(),
                nn.ConvTranspose2d(10, 1, (5, 5)),  # 1 x 28 x 28
                nn.Sigmoid(),
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)

            return x

    # Initialize the model, loss function, and optimizer
    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return criterion, model, optimizer


@app.cell
def _(criterion, model, optimizer, train_loader):
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        for _images, _ in train_loader:
            _reconstructed = model(_images)  # Forward pass
            _loss = criterion(_reconstructed, _images)
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()  # Backward pass and optimization
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {_loss.item():.4f}')
    return


@app.cell
def _(criterion, model, test_loader, torch):
    # Detect anomalies using reconstruction error
    threshold = 0.1  # Define a threshold for anomaly detection
    model.eval()
    anomalies = []
    with torch.no_grad():
        for _images, _ in test_loader:
            _reconstructed = model(_images)
            _loss = criterion(_reconstructed, _images)
            if _loss.item() > threshold:
                anomalies.append(_reconstructed)  # If reconstruction error exceeds the threshold, mark it as an anomaly
    return (anomalies,)


@app.cell
def _(anomalies, plt):
    # Visualize anomalies
    if anomalies:
        # Select the first anomaly and remove the channel dimension for visualization
        anomaly_image = anomalies[1][0].squeeze()  # Remove the channel dimension (1)
        print(f"Anomaly image shape: {anomaly_image.shape}")  # Optional: Check the shape of the image
        plt.imshow(anomaly_image.cpu().numpy(), cmap='gray')  # Convert tensor to NumPy array for visualization
        plt.show()
    else:
        print("No anomalies detected.")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
