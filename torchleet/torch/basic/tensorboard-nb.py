import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem: Visualize Training Progress with TensorBoard in PyTorch

    ### Problem Statement
    You are tasked with using **TensorBoard** to monitor the training progress of a linear regression model in PyTorch. TensorBoard provides a visual interface to track metrics such as loss during training, making it easier to analyze and debug your model.

    ### Requirements
    1. **TensorBoard Integration**:
       - Set up TensorBoard using `torch.utils.tensorboard.SummaryWriter`.
       - Log the training loss after each epoch.

    4. **Visualization**:
       - Start TensorBoard using the command:
         ```bash
         tensorboard --logdir=runs
         ```
       - Visualize the loss curve during training.

    ### Constraints
    - Ensure that TensorBoard logs are saved in a directory named `runs`.
    - The solution must handle multiple epochs and log the loss consistently.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    return SummaryWriter, nn, optim, torch


@app.cell
def _(nn, torch):
    # Generate synthetic data
    torch.manual_seed(42)
    X = torch.rand(100, 1) * 10  # 100 data points between 0 and 10
    y = 3 * X + 5 + torch.randn(100, 1)  # Linear relationship with noise

    # Define a simple Linear Regression Model
    class LinearRegressionModel(nn.Module):
        def __init__(self):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(1, 1)  # Single input and single output

        def forward(self, x):
            return self.linear(x)
    return LinearRegressionModel, X, y


@app.cell
def _(LinearRegressionModel, SummaryWriter, X, nn, optim, y):
    # TODO: Initialize SummaryWriter to log data
    writer = SummaryWriter()

    # Initialize the model, loss function, and optimizer
    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log loss to TensorBoard
        # TODO: Add the loss value to TensorBoard
        writer.add_scalar("loss", loss.item(), epoch)

        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Close the TensorBoard writer
    writer.close()

    # Run TensorBoard using the logs generated
    # Command to run: tensorboard --logdir=runs
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
