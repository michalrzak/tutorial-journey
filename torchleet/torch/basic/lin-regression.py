import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem: Implement Linear Regression

    ### Problem Statement
    Your task is to implement a **Linear Regression** model using PyTorch. The model should predict a continuous target variable based on a given set of input features.

    ### Requirements
    1. **Model Definition**:
       - Implement a class `LinearRegressionModel` with:
         - A single linear layer mapping input features to the target variable.
    2. **Forward Method**:
       - Implement the `forward` method to compute predictions given input data.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    return nn, optim, torch


@app.cell
def _(nn, optim, torch):
    # Generate synthetic data
    torch.manual_seed(42)
    X = torch.rand(100, 1) * 10  # 100 data points between 0 and 10
    y = 2 * X + 3 + torch.randn(100, 1)  # Linear relationship with noise

    # Define the Linear Regression Model
    #TODO: Add the layer and forward implementation
    class LinearRegressionModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            x = self.linear(x)
            return x
        
    # Initialize the model, loss function, and optimizer
    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs = 1000
    # Training loop
    for epoch in range(epochs):
        _predictions = model(X)
        loss = criterion(_predictions, y)  # Forward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:  # Backward pass and optimization
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')  # Log progress every 100 epochs
    return (model,)


@app.cell
def _(model, torch):
    # Display the learned parameters
    [w, b] = model.linear.parameters()
    print(f'Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}')
    X_test = torch.tensor([[4.0], [7.0]])
    # Testing on new data
    with torch.no_grad():
        _predictions = model(X_test)
        print(f'Predictions for {X_test.tolist()}: {_predictions.tolist()}')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
