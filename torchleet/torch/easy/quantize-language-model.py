import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Problem: Quantize Your Language Model

    ### Problem Statement
    Implement a **language model** using an LSTM and apply **dynamic quantization** to optimize it for inference. Dynamic quantization reduces the model size and enhances inference speed by quantizing the weights of the model.

    ### Requirements

    1. **Define the Language Model**:
       - **Purpose**: Build a simple language model that predicts the next token in a sequence.
       - **Components**:
         - **Embedding Layer**: Converts input tokens into dense vector representations.
         - **LSTM Layer**: Processes the embedded sequence to capture temporal dependencies.
         - **Fully Connected Layer**: Outputs predictions for the next token.
         - **Softmax Layer**: Applies a probability distribution over the vocabulary for predictions.
       - **Forward Pass**:
         - Pass the input sequence through the embedding layer.
         - Feed the embedded sequence into the LSTM.
         - Use the final hidden state from the LSTM to make predictions via the fully connected layer.
         - Apply the softmax function to obtain probabilities over the vocabulary.

    2. **Apply Dynamic Quantization**:
       - Quantize the model dynamically
       - Evaluate the quantized model's performance compared to the original model.
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.quantization.quantize import quantize
    return nn, optim, torch


@app.cell
def _(nn):
    # TODO: Define a simple Language Model (an LSTM-based model)
    class LanguageModel(nn.Module):
        def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
            super(LanguageModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.lstm = nn.LSTM(
                embed_size,
                hidden_size,
                num_layers,
                batch_first=True
            )
            self.final = nn.Linear(hidden_size, vocab_size)
            self.final_act = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.lstm(x)
            x = self.final(x[:, -1, :])
            x = self.final_act(x)

            return x
    return (LanguageModel,)


@app.cell
def _(LanguageModel, nn, optim, torch):
    # Create synthetic training data
    torch.manual_seed(42)
    vocab_size = 50
    seq_length = 10
    batch_size = 32
    X_train = torch.randint(0, vocab_size, (batch_size, seq_length))  # Random integer input
    y_train = torch.randint(0, vocab_size, (batch_size,))  # Random target words

    # Initialize the model, loss function, and optimizer
    embed_size = 64
    hidden_size = 128
    num_layers = 2
    model = LanguageModel(vocab_size, embed_size, hidden_size, num_layers)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return (
        X_train,
        criterion,
        model,
        optimizer,
        seq_length,
        vocab_size,
        y_train,
    )


@app.cell
def _(X_train, criterion, model, optimizer, y_train):
    # Training loop
    epochs = 5
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        # Log progress every epoch
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}")

    # Now, we will quantize the model dynamically to reduce its size and improve inference speed
    # Quantization: Apply dynamic quantization to the language model
    # quantized_model = quantize(model, {nn.Linear, nn.LSTM}, dtype=torch.qint8)

    # # Save the quantized model
    # torch.save(quantized_model.state_dict(), "quantized_language_model.pth")
    return


@app.cell
def _():
    # Load the quantized model and test it
    # quantized_model_1 = LanguageModel(vocab_size, embed_size, hidden_size, num_layers)
    # #quantized_model_1 = quantize_dynamic(quantized_model_1, {nn.Linear, nn.LSTM}, dtype=torch.qint8)
    # # Apply dynamic quantization on the model after defining it
    # quantized_model_1.load_state_dict(torch.load('quantized_language_model.pth'))
    return


@app.cell
def _(model, quantized_model_1, seq_length, torch, vocab_size):
    # Testing the quantized model on a sample input
    model.eval()
    test_input = torch.randint(0, vocab_size, (1, seq_length))
    with torch.no_grad():
        prediction = quantized_model_1(test_input)
        print(f'Prediction for input {test_input.tolist()}: {prediction.argmax(dim=1).item()}')
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
