import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Problem: Train a 3D CNN network for segmenting CT images
    ### Problem Statement
    You are tasked with employing and evaluating a 3D CNN model in Pytorch for semantic segmentation on synthetically generated CT images.
    Your goal is to review the input and label data shapes. Next, define a MedCNN model class with a `forward` method that emulates a encode-decoder architecture with appropriate input and output channels based on the input shapes.

    ### Requirements
    1. **Implement** a MedCNN model class with Conv3D and ConvTranspose3d for downsampling and upsampling respectively.
    2. **Define** Dice loss for the problem.
    2. **Perform** transfer learning from a ResNet18 - a common strategy for custom architectures.
    3. **Train** the model for 5 epochs.
    ### Constraints
    - Use `Pytorch` in-built convolution layers
    - Ensure, there is a segmentation head at the end of the network


    <details>
      <summary>💡 Hint</summary>
      - Strip off the `Avgpooling` and linear layers from ResNet18 using `list(resnet_model.children())[:-2]`
      <br>
      - [Conv3D](https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html)
      <br>
      - [ConvTranspose3D](https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose3d.html)
      <br>
      - [Forum discussion on model.children](https://discuss.pytorch.org/t/module-children-vs-module-modules/4551)
    </details>
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    return nn, optim, torch, torchvision


@app.cell
def _(torch):
    # Generate synthetic CT-scan data (batches, slices, RGB) and associated segmentation masks
    torch.manual_seed(42)
    batch = 100
    num_slices = 10
    channels = 3
    width = 256
    height = 256

    ct_images = torch.randn(size=(batch, num_slices, channels, width, height))
    segmentation_masks = (torch.randn(size=(batch, num_slices, 1, width, height))>0).float()

    print(f"CT images (train examples) shape: {ct_images.shape}")
    print(f"Segmentation binary masks (labels) shape: {segmentation_masks.shape}")
    return ct_images, segmentation_masks


@app.cell
def _(nn, torch):
    # Define the MedCNN class and its forward method
    class MedCNN(nn.Module):
        def __init__(self, backbone, out_channel=1):
            super(MedCNN, self).__init__()
            self.backbone = backbone

            # TODO: Add Downsample convolutional layers
            self.conv1 = nn.Conv3d(512, 64, (3, 3, 3), padding=1)
            self.conv2 = nn.Conv3d(64, 64, (3, 3, 3), padding=1)

            # TODO: Add Upsample convolutional layers
            self.convt1 = nn.ConvTranspose3d(64, 32, kernel_size=(1, 4, 4), stride=(1, 4, 4))
            self.convt2 = nn.ConvTranspose3d(32, 16, kernel_size=(1, 8, 8), stride=(1, 8, 8))

            #TODO: Final convolution layer from 16 to 1 channel
            self.convf = nn.Conv3d(16, 1, kernel_size=(3, 3, 3), padding=1)

            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            b, d, c, w, h = x.size() #Input size: [B, D, C, W, H]
            print(f"Input shape [B, D, C, W, H]: {b, d, c, w, h}")

            #TODO: make changes to the shape of the input such that it is compatible with ResNet
            # need to combine B and D dimension into one I suppose
            x = torch.reshape(x, [b*d, c, w, h])
            x = self.backbone(x) # b*d x 512 x 8 x 8
            print(x.shape)

            #TODO: take output features from the backbone ResNet and make it compatible with Conv3D format
            x = torch.reshape(x, [b, d, 512, 8, 8])
            x = torch.permute(x, [0, 2, 1, 3, 4])  # b x 512 x d x 8 x 8
        
            #TODO: Downsampling
            x = self.conv1(x)  # b x 64 x 10 x 8 x 8
            x = self.relu(x)
            x = self.conv2(x)  # b x 64 x 10 x 8 x 8
            x = self.relu(x)

            #TODO: Upsampling
            x = self.convt1(x) # b x 32 x 10 x 32 x 32
            x = self.relu(x)
            x = self.convt2(x) # b x 1 x 10 x 256 x 256
        

            #TODO: final segmentation head
            x = self.convf(x)
            x = self.sigmoid(x)

            return x
    return (MedCNN,)


@app.cell
def _(torch):
    #TODO: define Dice loss
    def compute_dice_loss(pred, labels, eps=1e-8):
        '''
        Args
        pred: [B, D, 1, W, H]
        labels: [B, D, 1, W, H]

        Returns
        dice_loss: float
        '''
        overlap = torch.sum(pred * labels)
        loss = 2 * overlap / (torch.sum(pred) + torch.sum(labels) + eps)
        print(loss)
        return loss
    return (compute_dice_loss,)


@app.cell
def _(MedCNN, nn, optim, torchvision):
    # Define resnet as the backbone removing the last two layers
    resnet_model = torchvision.models.resnet18(pretrained=True)
    resnet_model = nn.Sequential(*list(resnet_model.children())[:-2])

    model = MedCNN(backbone=resnet_model)

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    return model, optimizer, resnet_model


@app.cell
def _(compute_dice_loss, ct_images, model, optimizer, segmentation_masks):
    epochs = 5
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(ct_images)
        loss = compute_dice_loss(pred, segmentation_masks)
        loss.backward()
        optimizer.step()
        print(f"Loss at epoch {epoch}: {loss}")
    return


@app.cell
def _(resnet_model):
    resnet_model
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
