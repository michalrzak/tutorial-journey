import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Problem: Implement an RNN in PyTorch

    ### Problem Statement
    You are tasked with implementing a **Recurrent Neural Network (RNN)** in PyTorch to process sequential data. The model should contain an RNN layer for handling sequential input and a fully connected layer to output the final predictions. Your goal is to complete the RNN model by defining the necessary layers and implementing the forward pass.

    ### Requirements
    1. **Define the RNN Model**:
       - Add an **RNN layer** to process sequential data.
       - Add a **fully connected layer** to map the RNN output to the final prediction.

    ### Constraints
    - Use appropriate configurations for the RNN layer, including hidden units and input/output sizes.


    <details>
      <summary>💡 Hint</summary>
      Add the RNN layer (self.rnn) and fully connected layer (self.fc) in RNNModel.__init__.
      <br>
      Implement the forward pass to process inputs through the RNN layer and fully connected layer.
    </details>
    """)
    return


@app.cell
def _():
    # Generate synthetic sequential data
    torch.manual_seed(42)
    sequence_length = 10
    num_samples = 100

    # Create a sine wave dataset
    X = torch.linspace(0, 4 * 3.14159, steps=num_samples).unsqueeze(1)
    y = torch.sin(X)

    # Prepare data for RNN
    def create_in_out_sequences(data, seq_length):
        in_seq = []
        out_seq = []
        for i in range(len(data) - seq_length):
            in_seq.append(data[i:i + seq_length])
            out_seq.append(data[i + seq_length])
        return torch.stack(in_seq), torch.stack(out_seq)

    X_seq, y_seq = create_in_out_sequences(y, sequence_length)
    return X_seq, y_seq


@app.cell
def _(X_seq):
    print(X_seq)
    return


@app.class_definition
# Define the RNN Model
# TODO: Add RNN layer, fully connected layer, and forward implementation
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.W_t = nn.Parameter(torch.rand(1, 10)) # nn.Linear(1, 10) #nn.RNN(1, 10, 3)
        self.W_t_1 = nn.Parameter(torch.rand(10, 10))
        self.activation = nn.Tanh()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        if len(x.shape) == 2:
            l, dim = x.shape
        else:
            _, l, dim = x.shape

        x_rnn = torch.zeros_like(x[:, 0])
        prev = torch.zeros((1, 10))
        for i in range(l):
            x_rnn = x[:, i] @ self.W_t + prev
            x_rnn = self.activation(x_rnn)

            prev = x_rnn @ self.W_t_1  

        x = self.activation(x_rnn)
        x = self.linear(x)

        return x


@app.cell
def _(X_seq, y_seq):
    # Initialize the model, loss function, and optimizer
    model = RNNModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 500
    for epoch in range(epochs):
        for sequences, labels in zip(X_seq, y_seq):
            sequences = sequences.unsqueeze(0)  # Add batch dimension
            labels = labels.unsqueeze(0)  # Add batch dimension

            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    return (model,)


@app.cell
def _(model):
    # Testing on new data
    X_test = torch.linspace(4 * 3.14159, 5 * 3.14159, steps=10).unsqueeze(1)

    # Reshape to (batch_size, sequence_length, input_size)
    X_test = X_test.unsqueeze(0)  # Add batch dimension, shape becomes (1, 10, 1)

    with torch.no_grad():
        predictions = model(X_test)
        print(f"Predictions for new sequence: {predictions.tolist()}")
    return


if __name__ == "__main__":
    app.run()
