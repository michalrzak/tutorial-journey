import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem: Implement an LSTM Model

    ### Problem Statement
    You are tasked with implementing a simple **LSTM (Long Short-Term Memory)** model in PyTorch. The model should process sequential data using an LSTM layer followed by a fully connected (FC) layer. Your goal is two-fold: one is to implement a LSTM layer from scratch and another using inbuilt pytorch LSTM layer. Compare the results implementing the forward passes for both the LSTM models.

    ### Requirements
    1. **Define the LSTM Model using Custom LSTM layer**:
       - Add a `Custom` LSTM layer to the model. The layer must take care of the hidden and cell states
       - Add a **fully connected (FC) layer** that maps the output of the LSTM to the final predictions.
       - Implement the `forward` method to:
         - Pass the input sequence through the LSTM.
         - Feed the output of the LSTM into the fully connected layer for the final output.

    2. **Define the LSTM Model using in-built LSTM layer**:
      - Same as `1` with only difference that this time define the LSTM layer using pytorch `nn.Module`

    ### Constraints
    - The LSTM layer should be implemented with a single hidden layer.
    - Use a suitable number of input features, hidden units, and output size for the task.
    - Make sure the `forward` method returns the output of the fully connected layer after processing the LSTM output.


    <details>
      <summary>💡 Hint</summary>
      Add the LSTM layer and FC layer in LSTMModel.__init__.
      <br>
      Implement the forward pass to process sequences using the LSTM and FC layers.
      <br> Review Hidden and cell states computation here: [D2l.ai](https://d2l.ai/chapter_recurrent-modern/lstm.html)
    </details>
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    return nn, optim, plt, torch


@app.cell
def _(torch):
    # Generate synthetic sequential data
    torch.manual_seed(42)
    sequence_length = 10
    num_samples = 100

    # Create a sine wave dataset
    X = torch.linspace(0, 4 * 3.14159, steps=num_samples).unsqueeze(1)
    y = torch.sin(X)

    # Prepare data for LSTM
    def create_in_out_sequences(data, seq_length):
        in_seq = []
        out_seq = []
        for i in range(len(data) - seq_length):
            in_seq.append(data[i:i + seq_length])
            out_seq.append(data[i + seq_length])
        return torch.stack(in_seq), torch.stack(out_seq)

    X_seq, y_seq = create_in_out_sequences(y, sequence_length)
    return X_seq, create_in_out_sequences, sequence_length, y_seq


@app.cell
def _(nn, torch):
    class CustomLSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_units):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_units = hidden_units

            self.W_xi = nn.Parameter(torch.rand((input_dim, hidden_units)))  # input
            self.W_hi = nn.Parameter(torch.rand((hidden_units, hidden_units)))
            self.b_i = nn.Parameter(torch.rand(hidden_units))
            self.act_i = nn.Sigmoid()
            self.W_xf = nn.Parameter(torch.rand((input_dim, hidden_units)))  # forget
            self.W_hf = nn.Parameter(torch.rand((hidden_units, hidden_units)))
            self.b_f = nn.Parameter(torch.rand(hidden_units))
            self.act_f = nn.Sigmoid()
            self.W_xo = nn.Parameter(torch.rand((input_dim, hidden_units)))  # output
            self.W_ho = nn.Parameter(torch.rand((hidden_units, hidden_units)))
            self.b_o = nn.Parameter(torch.rand(hidden_units))
            self.act_o = nn.Sigmoid()
            self.W_xc = nn.Parameter(torch.rand((input_dim, hidden_units)))  # input node
            self.W_hc = nn.Parameter(torch.rand((hidden_units, hidden_units)))
            self.b_c = nn.Parameter(torch.rand(hidden_units))
            self.act_c = nn.Tanh()

            self.act_hidden = nn.Tanh()

            self.fc = nn.Linear(hidden_units, 1)

        
        def forward(self, inputs, H_C=None):
            if H_C is None:
                H = torch.zeros((inputs.shape[0], self.hidden_units))
                C = torch.zeros((inputs.shape[0], self.hidden_units))

            outputs = []
            for i in range(inputs.shape[1]):
                X = inputs[:, i, :]
                I = self.act_i(X @ self.W_xi + H @ self.W_hi + self.b_i)
                F = self.act_f(X @ self.W_xf + H @ self.W_hf + self.b_f)
                O = self.act_o(X @ self.W_xo + H @ self.W_ho + self.b_o)
                C_tilde = self.act_c(X @ self.W_xc + H @ self.W_hc + self.b_c)
    
                C = F * C + I * C_tilde
                H = O * self.act_hidden(C)

                outputs.append(H.unsqueeze(1))

        
            # return outputs, (H, C)
            outputs = torch.cat(outputs, dim=1)
            pred = self.fc(outputs)
            return pred, (H, C)
    
    return (CustomLSTMModel,)


@app.cell
def _(X_seq):
    X_seq.shape
    return


@app.cell
def _(nn, optim):
    # Define the LSTM Model
    # TODO: Add LSTM layer, forward implementation
    class LSTMModel(nn.Module):
        def __init__(self, input_dim, hidden_cells):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_cells, batch_first=True)
            self.fc = nn.Linear(hidden_cells, 1)

        def forward(self, x):
            outputs, (H, C) = self.lstm(x)
            pred = self.fc(outputs)
            return self.fc(outputs)
        
    # Initialize the model, loss function, and optimizer
    model = LSTMModel(1, 50)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    return (LSTMModel,)


@app.cell
def _(CustomLSTMModel, LSTMModel, nn, optim):
    # Initialize the model, loss function, and optimizer
    model_custom = CustomLSTMModel(1, 50)
    model_inbuilt = LSTMModel(1, 50)
    criterion_1 = nn.MSELoss()
    optimizer_custom = optim.Adam(model_custom.parameters(), lr=0.01)
    optimizer_inbuilt = optim.Adam(model_inbuilt.parameters(), lr=0.01)
    return (
        criterion_1,
        model_custom,
        model_inbuilt,
        optimizer_custom,
        optimizer_inbuilt,
    )


@app.cell
def _(X_seq, criterion_1, model_custom, optimizer_custom, y_seq):
    _epochs = 500
    for _epoch in range(_epochs):
        state = None
        _pred, state = model_custom(X_seq, state)
        _loss = criterion_1(_pred[:, -1, :], y_seq)
        optimizer_custom.zero_grad()
        _loss.backward()
        optimizer_custom.step()
        if (_epoch + 1) % 50 == 0:
            print(f'Epoch [{_epoch + 1}/{_epochs}], Loss: {_loss.item():.4f}')
    return


@app.cell
def _(X_seq, criterion_1, model_inbuilt, optimizer_inbuilt, y_seq):
    _epochs = 500
    for _epoch in range(_epochs):
        _pred = model_inbuilt(X_seq)
        _loss = criterion_1(_pred[:, -1, :], y_seq)
        optimizer_inbuilt.zero_grad()
        _loss.backward()
        optimizer_inbuilt.step()
        if (_epoch + 1) % 50 == 0:
            print(f'Epoch [{_epoch + 1}/{_epochs}], Loss: {_loss.item():.4f}')
    return


@app.cell
def _(
    create_in_out_sequences,
    model_custom,
    model_inbuilt,
    sequence_length,
    torch,
):
    # Testing on new data
    test_steps = 100  # Ensure this is greater than sequence_length
    X_test = torch.linspace(0, 5 * 3.14159, steps=test_steps).unsqueeze(1)
    y_test = torch.sin(X_test)

    # Create test input sequences
    X_test_seq, _ = create_in_out_sequences(y_test, sequence_length)

    with torch.no_grad():
        pred_custom, _ = model_custom(X_test_seq)
        pred_inbuilt = model_inbuilt(X_test_seq)
    pred_custom = torch.flatten(pred_custom[:, -1, :])
    pred_inbuilt = torch.flatten(pred_inbuilt[:, -1, :])
    print(f"Predictions with Custom Model for new sequence: {pred_custom.tolist()}")
    print(f"Predictions with In-Built Model: {pred_inbuilt.tolist()}")
    return pred_custom, pred_inbuilt


@app.cell
def _(plt, pred_custom, pred_inbuilt):
    #Plot the predictions
    plt.figure()
    # plt.plot(y_test, label="Ground Truth")
    plt.plot(pred_custom, label="custom model")
    plt.plot(pred_inbuilt, label="inbuilt model")
    plt.legend()
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
