import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")

with app.setup:
    import torch
    import torchvision
    import torchvision.transforms.v2 as transforms
    import torch.utils.data as data
    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Problem: Use torchvision.transforms for Data Augmentation

    ### Problem Statement
    You are tasked with applying data augmentation techniques to image data using **torchvision.transforms**. The goal is to enhance the variability of the input data by applying the following transformations:
    - **Random Horizontal Flip**: Flip the image horizontally with a probability of 0.5.
    - **Random Crop**: Randomly crop the image to a specific size.
    - **Normalization**: Normalize the image using a specified mean and standard deviation.

    Additionally, you will visualize a batch of augmented images to verify the transformations have been applied correctly.

    ### Requirements
    1. **Apply Data Augmentation**:
       - Use **torchvision.transforms** to apply:
         - **Random Horizontal Flip**: Flip the image randomly with a probability of 0.5.
         - **Random Crop**: Randomly crop the image to a size of 28x28.
         - **Normalization**: Normalize the image with specified mean and standard deviation values.

    2. **Verify Augmentation**:
       - Visualize a batch of augmented images using **matplotlib** to ensure the transformations are applied correctly.

    ### Constraints
    - Ensure the transformations are applied in the correct order.
    - Normalize the images using appropriate values for mean and standard deviation, such as those used in common datasets like CIFAR-10.

    <details>
      <summary>💡 Hint</summary>
      Define the transforms.Compose pipeline:
      <br>
      Include RandomHorizontalFlip, RandomCrop, and Normalize in the correct sequence.
      <br>
      <br>
      Visualize the Augmentation:
      <br>
      Use imshow to confirm that the transformations are applied correctly.
    </details>
    """)
    return


@app.cell
def _():
    # Load CIFAR-10 dataset with data augmentation
    # TODO: Apply random horizontal flip, random crop, and normalization to the images
    transform = transforms.Compose([
        # transforms.ToImage(),
        #transforms.ToDtype(dtype=torch.float32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize([0., 0., 0.], [1., 1., 1.])
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    return (train_loader,)


@app.cell
def _(train_loader):
    # Display a batch of augmented images
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # Get some random training images
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # Show images
    imshow(torchvision.utils.make_grid(images))
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
