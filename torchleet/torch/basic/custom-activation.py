import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem: Write a Custom Activation Function

    ### Problem Statement
    You are tasked with implementing a **custom activation function** in PyTorch that computes the following operation:

    $$ \text{activation}(x) = \tanh(x) + x $$

    Once implemented, this custom activation function will be used in a simple linear regression model.

    ### Requirements
    1. **Custom Activation Function**:
       - Implement a class `CustomActivationModel` inheriting from `torch.nn.Module`.
       - Define the `forward` method to compute the activation function \( \text{tanh}(x) + x \).

    2. **Integration with Linear Regression**:
       - Use the custom activation function in a simple linear regression model.
       - The model should include:
         - A single linear layer.
         - The custom activation function applied to the output of the linear layer.

    ### Constraints
    - The custom activation function should not have any learnable parameters.
    - Ensure compatibility with PyTorch tensors for forward pass computations.

    <details>
      <summary>💡 Hint</summary>
      Some details: https://stackoverflow.com/questions/55765234/pytorch-custom-activation-functions
    </details>
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    return nn, optim, torch


@app.cell
def _(torch):
    # Generate synthetic data
    torch.manual_seed(42)
    X = torch.rand(100, 1) * 10  # 100 data points between 0 and 10
    y = 2 * X + 3 + torch.randn(100, 1)  # Linear relationship with noise
    return X, y


@app.cell
def _(X, nn, optim, torch, y):
    # Define the Linear Regression Model
    class CustomActivationModel(nn.Module):

        def __init__(self):
            super(CustomActivationModel, self).__init__()  # Single input and single output
            self.linear = nn.Linear(1, 1)
      # TODO: Implement the forward pass
        def custom_activation(self, x):
            return torch.tanh(x) + x

        def forward(self, x):  # TODO: Implement the forward pass
            x = self.linear(x)
            x = self.custom_activation(x)
            return x
        
    model = CustomActivationModel()
    criterion = nn.MSELoss()
    # Initialize the model, loss function, and optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs = 1000
    for epoch in range(epochs):
        _predictions = model(X)
    # Training loop
        loss = criterion(_predictions, y)
        optimizer.zero_grad()
        loss.backward()  # Forward pass
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')  # Backward pass and optimization  # Log progress every 100 epochs
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
