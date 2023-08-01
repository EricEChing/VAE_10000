import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataLoader import train_loader
from train_vae import *
from vae import *


def vae_loss(y_true, y_pred, mu, log_var):
    """
    Calculate the Variational Autoencoder (VAE) loss using Mean Squared Error (MSE) and Kullback-Leibler Divergence (KLD).

    Parameters:
        y_true (Tensor): The true input data.
        y_pred (Tensor): The predicted output data.
        mu (Tensor): The mean of the latent distribution.
        log_var (Tensor): The log-variance of the latent distribution.

    Returns:
        loss (Tensor): The VAE loss.
    """
    # Calculate MSE loss
    mse_loss = F.mse_loss(y_pred, y_true)

    # Calculate KLD loss
    kld_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    # Combine the losses
    loss = mse_loss + kld_loss

    return loss


def validation_Model(model, valid_loader, vae_loss, device):
    model.eval()
    val_loss = 0

    # Test model using validation data:
    with torch.inference_mode():
        for start, data in enumerate(valid_loader, 0):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            # Calculate the loss:
            loss = vae_loss(y_pred=recon_batch, y_true=data, mu=mu, log_var=logvar)
            val_loss += loss.item()

    return val_loss / len(valid_loader)


def test_model(model, test_loader, vae_loss, device):
    """
    Test the PyTorch model on the test dataset.

    Args:
        model (nn.Module): The PyTorch model to be tested.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion: The loss function used for evaluation.
        device (str): Device to run the model on (default is 'cpu').

    Returns:
        average_loss (float): Average loss on the test dataset.
        accuracy (float): Accuracy of the model on the test dataset.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.inference_mode():  # Disable gradient calculation during evaluation
        for start, data in enumerate(test_loader, 0):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            # Calculate the loss:
            loss = vae_loss(y_pred=recon_batch, y_true=data, mu=mu, log_var=logvar)
            total_loss += loss.item()

    average_loss = total_loss / len(test_loader)
    print(f"The test loss = {average_loss}")

    return average_loss
