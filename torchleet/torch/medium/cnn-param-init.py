import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem: Implement parameter initialization strategies for a CNN model in Pytorch

    ### Problem Statement
    You are tasked with employing and evaluating a CNN model\'s parameter initialization strategies in Pytorch.
    Your goal is to initialize the weights and biases of a vanilla CNN model provided in the problem statement and comment on the implications of each strategy.

    ### Requirements
    1. **Initialize** weights and biases in the following ways:
       - **Zero Initialization**: set the parameters to zero
       - **Random Initialization**: sets model parameters to random values drawn from a normal distribution
       - **Xavier Initialization** sets them to random values from a normal distribution with **mean=0 and variance=1\/n**
       - **Kaiming He Initialization** initializes to random values from a normal distribution with **mean=0 and variance=2\/n**
    2. Train and compute accuracy for each strategy
    ### Constraints
    - Use the given CNN model and the training and testing helper functions for accuracy computations.
    - Ensure the model is compatible with the CIFAR-10 dataset, which contains 10 classes.


    <details>
      <summary>💡 Hint</summary>
      - Use `torch.nn.init` for weight initialization
      <br>
      - Resources to read: [All you need is a good init](https://arxiv.org/pdf/1511.06422)
    </details>
    """)
    return


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    return mo, nn, optim, torch, torchvision, transforms


@app.cell
def _(torch, torchvision, transforms):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
    return test_loader, train_loader


@app.cell
def _(nn, optim, torch):
    def train_test_loop(model, train_loader, test_loader, epochs=10):
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            for image, label in train_loader:
                pred = model(image)
                loss = criterion(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    
            print(f"Training loss at epoch {epoch} = {loss.item()}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for image_test, label_test in test_loader:
                pred_test = model(image_test)
                _, pred_test_vals = torch.max(pred_test, dim=1)
                total += label_test.size(0)
                correct += (pred_test_vals == label_test).sum().item()
        print(f"Test Accuracy = {(correct * 100)/total}")
    return (train_test_loop,)


@app.cell
def _(nn):
    class VanillaCNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*16*16, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    return (VanillaCNNModel,)


@app.cell
def _(nn):
    def config_init(init_type="kaiming"):
        # TODO: Add Kaiming initialization according to a CNN or a Linear layer
        def kaiming_init(m):
            try:
                nn.init.kaiming_normal_(m.weight)
            except AttributeError:
                return
    
        # TODO: Implement Xavier initialization        
        def xavier_init(m):
            try:
                return nn.init.xavier_normal_(m.weight)
            except AttributeError:
                return

        # TODO: Initialize weights and biases to zero          
        def zeros_init(m):
            try:
                return nn.init.zeros_(m.weight)
            except AttributeError:
                return

        # TODO: Initialize weights and biases from a normal distribution          
        def random_init(m):
            try:
                return nn.init.normal_(m)
            except AttributeError:
                return


        initializer_dict = {"kaiming": kaiming_init,
                            "xavier": xavier_init,
                            "zeros": zeros_init,
                            "random": random_init}

        return initializer_dict.get(init_type)
    return (config_init,)


@app.cell
def _(
    VanillaCNNModel,
    config_init,
    test_loader,
    train_loader,
    train_test_loop,
):
    for name, model in zip(["Vanilla", "Kaiming", "Xavier", "Zeros", "Random"], [VanillaCNNModel(),
                  VanillaCNNModel().apply(config_init("kaiming")),
                  VanillaCNNModel().apply(config_init("xavier")),
                  VanillaCNNModel().apply(config_init("zeros")),
                  VanillaCNNModel().apply(config_init("random"))
                  ]):
        print(f"_________{name}_______________________")
        train_test_loop(model, train_loader, test_loader)
    return


@app.cell
def _(VanillaCNNModel):
    next(VanillaCNNModel().children()).weight
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
