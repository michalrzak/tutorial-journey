import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem: Write a Custom Activation Function with Autograd

    ### Problem Statement
    Implement a **custom activation function**, **Learned-SiLU**, using `torch.autograd.Function`. The activation function should be based on the SiLU formula \( x \cdot \text{sigmoid}(x) \) but include a **learnable slope parameter**. Use this custom activation function in a simple linear regression model.

    ### Requirements
    1. **Define the Custom Activation Function**:
       - Implement a custom activation function, **Learned-SiLU**, where the output is calculated as:
       - The **slope** should be a learnable parameter.

    $$\text{Learned-SiLU}(x) = \text{slope} \cdot x \cdot \text{sigmoid}(x)$$


    2. **Autograd Implementation**:
       - Use `torch.autograd.Function` to define the forward and backward passes for the custom activation function.

    3. **Integrate the Activation Function**:
       - Incorporate the custom activation function into a simple linear regression model.
       - Train the model to verify the functionality of the activation function.

    ### Constraints
    - Ensure the **slope parameter** is properly initialized and updated during training.

    <details>
      <summary>💡 Hint</summary>
      Some details: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    </details>


    <details>
      <summary>💡 Alternate Implementation?</summary>
      Can be done with nn.Module without implementing backward.
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
    class LearnedSiLUFunction(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, slope):  # Save the input tensor and slope for backward computation
            ctx.save_for_backward(x, slope)
            return slope * x * torch.sigmoid(x)

        @staticmethod
        def backward(ctx, grad_output):
            x, slope = ctx.saved_tensors
            sigmoid_x = torch.sigmoid(x)

            # w.r.t slope
            grad_slope = (grad_output * x * sigmoid_x).mean().view_as(slope)
        
            # w.r.t x
            grad_x = grad_output * (slope * sigmoid_x + slope * x * sigmoid_x * (1 - sigmoid_x))
            return grad_x, grad_slope

    class LinearRegressionModel(nn.Module):

    # Define the Linear Regression Model
        def __init__(self, slope=1):
            super().__init__()
            self.slope = nn.Parameter(torch.ones(1) * slope)
            self.weight = nn.Linear(1, 1)

        def forward(self, x):
            x = self.weight(x)
            x = LearnedSiLUFunction.apply(x, self.slope)
            return x
        
    model = LinearRegressionModel()  # Use the custom LearnedSiLUFunction
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # Initialize the model, loss function, and optimizer
    epochs = 1000
    for epoch in range(epochs):
        _predictions = model(X)
        loss = criterion(_predictions, y)
        # Training loop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # Forward pass
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')  # Backward pass and optimization  # Log progress every 100 epochs
    return (model,)


@app.cell
def _(model, torch):
    # Display the learned parameters
    [w, b] = model.weight.parameters()
    s = model.slope
    print(f'Learned weight: {w.item():.4f}, Learned bias: {b.item():.4f}, Learned slope: {s.item():.4f}')
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
