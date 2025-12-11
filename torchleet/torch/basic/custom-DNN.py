import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem: Implement a Deep Neural Network

    ### Problem Statement
    You are tasked with constructing a **Deep Neural Network (DNN)** model to solve a regression task using PyTorch. The objective is to predict target values from synthetic data exhibiting a non-linear relationship.

    ### Requirements
    Implement the `DNNModel` class that satisfies the following criteria:

    1. **Model Definition**:
       - The model should have:
         - An **input layer** connected to a **hidden layer**.
         - A **ReLU activation function** for non-linearity.
         - An **output layer** with a single unit for regression.

    <details> <summary>💡 Hint</summary> - Use `nn.Sequential` to simplify the implementation of the `DNNModel`. - Experiment with different numbers of layers and hidden units to optimize performance. - Ensure the final layer has a single output unit (since it's a regression task). </details> <details> <summary>💡 Bonus: Try Custom Loss Functions</summary> Experiment with custom loss functions (e.g., Huber Loss) and compare their performance with MSE. </details>
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.optim as optim
    return mo, nn, optim, torch


@app.cell
def _(torch):
    # Generate synthetic data
    torch.manual_seed(42)
    X = torch.rand(100, 2) * 10  # 100 data points with 2 features
    y = (X[:, 0] + X[:, 1] * 2).unsqueeze(1) + torch.randn(100, 1)  # Non-linear relationship with noise
    return X, y


@app.cell
def _(nn):
    # Define the Deep Neural Network Model
    class DNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layer = nn.Linear(2, 10)
            self.activation = nn.ReLU()
            self.output_layer = nn.Linear(10, 1)

        def forward(self, x):
            x = self.input_layer(x)
            x = self.activation(x)
            x = self.output_layer(x)
            return x
    return (DNNModel,)


@app.cell
def _(DNNModel, X, nn, optim, torch, y):
    # Initialize the model, loss function, and optimizer
    model = DNNModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        # Forward pass
        predictions = model(X)
        loss = criterion(predictions, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Testing on new data
    X_test = torch.tensor([[4.0, 3.0], [7.0, 8.0]])
    with torch.no_grad():
        predictions = model(X_test)
        print(f"Predictions for {X_test.tolist()}: {predictions.tolist()}")
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
