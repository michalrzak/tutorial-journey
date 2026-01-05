import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem: Implement Mixed Precision Training Using `torch.cuda.amp`

    ### Problem Statement
    Mixed precision training uses both 16-bit and 32-bit floating-point types to accelerate training and reduce memory usage without significantly affecting model performance. Your task is to implement mixed precision training for a deep learning model using PyTorch's `torch.cuda.amp`.

    ### Requirements

    1. **Enable Mixed Precision Training**:
       - Context manager to enable mixed precision for the forward pass.
       - Scale gradients during backpropagation and ensure stability.
    """)
    return


@app.cell
def _():
    # Implement mixed precision training in PyTorch using torch.cuda.amp
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    return DataLoader, TensorDataset, nn, optim, torch


@app.cell
def _(torch):
    torch.amp.autocast_mode.is_autocast_available("")
    return


@app.cell
def _(DataLoader, TensorDataset, nn, optim, torch):
    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    # Generate synthetic data
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = SimpleModel()  # .cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Enable mixed precision training
    scaler = torch.amp.GradScaler("cpu")
    return criterion, dataloader, model, optimizer, scaler


@app.cell
def _(criterion, dataloader, model, optimizer, scaler, torch):
    # Training loop
    epochs = 5
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs, labels

            # Forward pass under autocast
            with torch.amp.autocast("cpu"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass with scaled gradients
            optimizer.zero_grad()
            # TODO: Set scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
    return


@app.cell
def _(model, torch):
    # Test the model on new data
    X_test = torch.randn(5, 10)
    with torch.no_grad(), torch.amp.autocast("cpu"):
        predictions = model(X_test)
        print("Predictions:", predictions)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
