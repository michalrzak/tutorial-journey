import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from copy import copy
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    return F, nn, optim, torch


@app.cell
def _(nn, torch):
    class Softmax(nn.Module):

        def __init__(self):
            super().__init__()

        # def forward(self, x):
        #     x = torch.tensor(
        #         [[torch.e ** ele2 for ele2 in ele1] for ele1 in x],
        #         device=x.device,
        #         requires_grad=x.requires_grad
        #     )

        #     x_sum = torch.sum(x, dim=-1)

        #     x = torch.tensor(
        #         [[ele2 / ele_sum for ele2 in ele] for ele, ele_sum in zip(x, x_sum)],
        #         device=x.device,
        #         requires_grad=x.requires_grad
        #     )
        #     return x

        def forward(self, x):
            x = torch.exp(x)
            x = x / torch.sum(x, dim=-1, keepdim=True)
            return x
    
    return (Softmax,)


@app.cell
def _(Softmax, nn):
    class SimpleModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 3)
            self.softmax = Softmax()

        def forward(self, x):
            x = self.linear(x)
            x = self.softmax(x)
            return x

    return (SimpleModel,)


@app.cell
def _(F, model, nn, optim, torch):
    X = torch.concat((torch.rand(50, 1), torch.rand(50, 1) + 1, torch.rand(50, 1) + 2))
    y = F.one_hot(
        torch.concat((torch.tensor([0] * 50), torch.tensor([1] * 50), torch.tensor([2] * 50)))
    ).float()
    epochs = 100
    torch.manual_seed(42)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    return X, criterion, epochs, optimizer, y


@app.cell
def _(y):
    y
    return


@app.cell
def _(SimpleModel):
    model = SimpleModel()
    return (model,)


@app.cell
def _(X, criterion, epochs, model, optimizer, y):
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_hat = model(X)
        loss = criterion(y, y_hat)
        loss.backward()
    return


@app.cell
def _(model, torch):
    model(torch.tensor([[0.2], [1.5], [2.5]]))
    return


@app.cell
def _(X):
    X

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
