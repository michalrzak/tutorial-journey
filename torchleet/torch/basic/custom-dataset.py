import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem: Write a custom Dataset and Dataloader to load from a CSV file

    ### Problem Statement
    You are tasked with creating a **custom Dataset** and **Dataloader** in PyTorch to load data from a given `data.csv` file. The loaded data will be used to run a pre-implemented linear regression model.

    ### Requirements
    1. **Dataset Class**:
       - Implement a class `CustomDataset` that:
         - Reads data from a provided `data.csv` file.
         - Stores the features (X) and target values (Y) separately.
         - Implements PyTorch's `__len__` and `__getitem__` methods for indexing.

    2. **Dataloader**:
       - Use PyTorch's `DataLoader` to create an iterable for batch loading the dataset.
       - Support user-defined batch sizes and shuffling of the data.
    """)
    return


@app.cell
def _():
    import torch
    import pandas as pd

    torch.manual_seed(42)
    X = torch.rand(100, 1) * 10  # 100 data points between 0 and 10
    y = 2 * X + 3 + torch.randn(100, 1)  # Linear relationship with noise

    # Save the generated data to data.csv
    data = torch.cat((X, y), dim=1)
    df = pd.DataFrame(data.numpy(), columns=['X', 'y'])
    df.to_csv('data.csv', index=False)
    return X, pd, torch, y


@app.cell
def _():
    import torch.nn as nn
    import torch.optim as optim
    return nn, optim


@app.cell
def _(X, pd, torch, y):
    from torch.utils.data import Dataset, DataLoader

    class LinearRegressionDataset(Dataset):

        def __init__(self, path):
            data = pd.read_csv(path)
            self.X = torch.tensor(data["X"].values, dtype=torch.float32).view(-1, 1)
            self.y = torch.tensor(data["y"].values, dtype=torch.float32).view(-1, 1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, index):
            return X[index], y[index]
        
    # TODO: Add the missing code
    dataset = LinearRegressionDataset('data.csv')
    # Example usage of the DataLoader
    dataloader = DataLoader(dataset)
    return (dataloader,)


@app.cell
def _(dataloader, nn, optim):
    # Define the Linear Regression Model
    class LinearRegressionModel(nn.Module):

        def __init__(self):
            super(LinearRegressionModel, self).__init__()  # Single input and single output
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)
    # Initialize the model, loss function, and optimizer
    model = LinearRegressionModel()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs = 1000
    # Training loop
    for epoch in range(epochs):
        for batch_X, batch_y in dataloader:
            _predictions = model(batch_X)
            loss = criterion(_predictions, batch_y)  # Forward pass
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
