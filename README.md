# CPC Code README
This project includes the implementation of Contrastive Predictive Coding (CPC) using PyTorch.

# Overview
The Contrastive Predictive Coding (CPC) is a self-supervised learning algorithm that learns useful representations of the input data by predicting future samples in a latent space. These representations can then be used downstream for various tasks such as classification.

In this specific implementation, a model class CPC is defined which inherits from nn.Module. The model contains two GRU networks, the first one to transform the input, and the second one to predict future samples in the transformed space. The forward pass of the model returns an accuracy metric, a contrastive loss (nce_loss), and the output of the final layer of the model after applying a linear transformation.

The code then goes on to define the training and validation loops. During training, the model parameters are updated to minimize a loss function, which is a sum of a standard cross-entropy loss and the contrastive loss from the CPC model. During validation, the model's performance is evaluated on the test data.

# Requirements
PyTorch
NumPy
tqdm
# Instructions
Define your train_dataloader and test_dataloader. These should yield pairs of (inputs, labels).

Set DEVICE to your preferred device. For example, if you want to use a GPU, you can set DEVICE = torch.device("cuda:0").

Instantiate your model CPC_model = CPC(input_layer, hidden_layer, n_layer, num_class, group, time_step) and set the model to the device CPC_model.to(DEVICE).

Set your optimizer (e.g., optimizer = torch.optim.Adam(CPC_model.parameters(), lr=0.001)).

Set the number of epochs for training EPOCH.

Run the training and validation loop.

# Output
During training and validation, the model's performance is printed out. The performance includes:

Epoch number
Training loss
Validation loss
Classification report
For each epoch, the model's loss on the training data and validation data is printed. At the end of each validation loop, a classification report is printed, showing precision, recall, F1-score, and support for each class and the accuracy of the model. The loss for each epoch is also appended to a list for further analysis or plotting.

Please ensure to set the required parameters in the model instantiation as well as adjust the dataloader according to your dataset.
