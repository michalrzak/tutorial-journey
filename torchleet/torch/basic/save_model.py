import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem: Save and Load Your PyTorch Model

    ### Problem Statement
    You are tasked with saving a trained PyTorch model to a file and later loading it for inference or further training. This process allows you to persist the trained model and use it in different environments without retraining.

    ### Requirements
    1. **Save the Model**:
       - Save the model’s **state dictionary** (weights) to a file named `model.pth` using `torch.save`.

    2. **Load the Model**:
       - Load the saved state dictionary into a new model instance using `torch.load`.
       - Verify that the loaded model works as expected by performing inference or testing.
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
    # Define a simple model
    class SimpleModel(nn.Module):

        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(1, 1)

        def forward(self, x):
            return self.fc(x)
    # Create and train the model
    torch.manual_seed(42)
    model = SimpleModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    X = torch.rand(100, 1)
    # Training loop
    y = 3 * X + 2 + torch.randn(100, 1) * 0.1
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        _predictions = model(X)
        loss = criterion(_predictions, y)
        loss.backward()
        optimizer.step()
    return SimpleModel, model


@app.cell
def _(SimpleModel, model, torch):
    # TODO: Save the model to a file named "model.pth"
    torch.save(model.state_dict(), "model.pth")

    # TODO: Load the model back from "model.pth"
    loaded_model = SimpleModel()
    loaded_model.load_state_dict(torch.load("model.pth")) 
    return (loaded_model,)


@app.cell
def _(loaded_model, torch):
    # Verify the model works after loading
    X_test = torch.tensor([[0.2], [1.0], [1.5]])
    with torch.no_grad():
        _predictions = loaded_model(X_test)
        print(f'Predictions after loading: {_predictions}')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
