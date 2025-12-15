import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem: Implement Custom Loss Function (Huber Loss)

    ### Problem Statement
    You are tasked with implementing the **Huber Loss** as a custom loss function in PyTorch. The Huber loss is a robust loss function used in regression tasks, less sensitive to outliers than Mean Squared Error (MSE). It transitions between L2 loss (squared error) and L1 loss (absolute error) based on a threshold parameter $\delta$.

    The Huber loss is mathematically defined as:

    $$
    L_{\delta}(y, \hat{y}) =
    \begin{cases}
    \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \leq \delta, \\
    \delta \cdot (|y - \hat{y}| - \frac{1}{2} \delta) & \text{for } |y - \hat{y}| > \delta,
    \end{cases}
    $$

    where:
    - $y$ is the true value,
    - $\hat{y}$ is the predicted value,
    - $\delta$ is a threshold parameter that controls the transition between L1 and L2 loss.

    ### Requirements
    1. **Custom Loss Function**:
       - Implement a class `HuberLoss` inheriting from `torch.nn.Module`.
       - Define the `forward` method to compute the Huber loss as per the formula.

    2. **Usage in a Regression Model**:
       - Integrate the custom loss function into a regression training pipeline.
       - Use it to compute and optimize the loss during model training.

    ### Constraints
    - The implementation must handle both scalar and batch inputs for $y$ (true values) and $\hat{y}$ (predicted values).


    Extra Details: https://en.wikipedia.org/wiki/Huber_loss

    <details>
      <summary>💡 Hint</summary>
      Some details: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
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
def _(nn, optim, torch):
    # Generate synthetic data
    torch.manual_seed(42)
    X = torch.rand(100, 1) * 10  # 100 data points between 0 and 10
    y = 2 * X + 3 + torch.randn(100, 1)  # Linear relationship with noise

    #TODO: Define the nn.Module for the Huber Loss
    class HuberLoss(nn.Module):

        def __init__(self, delta):
            super().__init__()
            self.delta = delta

        def forward(self, y, y_hat):        
            loss = torch.where(
                torch.abs(y - y_hat) <= self.delta,
                1 / 2 * (y - y_hat) ** 2,
                self.delta * (torch.abs(y - y_hat) - 1/2 * self.delta)
            )
            return torch.mean(loss)
    

    class LinearRegressionModel(nn.Module):
    # Define the Linear Regression Model

        def __init__(self):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(1, 1)  # Single input and single output

        def forward(self, x):
            return self.linear(x)
        
    model = LinearRegressionModel()
    # Initialize the model, loss function, and optimizer
    criterion = HuberLoss(0.5)
    #TODO: Add the loss 
    optimizer = optim.SGD(model.parameters(), lr=0.01) 
    epochs = 1000
    for epoch in range(epochs):
    # Training loop
        _predictions = model(X)
        loss = criterion(_predictions, y)
        optimizer.zero_grad()  # Forward pass
        loss.backward()
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
