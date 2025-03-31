import torch
from torch import nn
from torch.nn import functional as nnf


class VAELoss(nn.Module):
    def __init__(self, smooth_factor=0.99):
        super(VAELoss, self).__init__()
        self.smooth_factor = smooth_factor  # Weighting for moving average
        self.mse_avg = None  # Moving average of MSE
        self.kl_avg = None  # Moving average of KL

    def get_mse(self, x, out):
        """Compute MSE loss."""
        x_expanded = x.unsqueeze(0).expand_as(out)
        return torch.mean((out - x_expanded) ** 2)

    def forward(self, x, out, z_mean, z_log_var, kl_factor=None):
        """
        Compute VAE loss with an adaptive beta scaling for KL divergence.

        Parameters:
        - x: Ground truth input
        - out: Reconstructed output from the decoder
        - z_mean: Mean vector from the encoder
        - z_log_var: Log variance vector from the encoder
        - kl_factor: Fixed coefficient for KL loss (if provided, dynamic beta is ignored)

        Returns:
        - loss: Total VAE loss
        - mse_loss: MSE component
        - kl_loss: KL divergence component
        - beta_value: Current value of beta (scaling factor for KL)
        """
        mse_loss = self.get_mse(x, out)

        # Compute KL divergence
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
        kl_loss = kl_loss.mean()  # Average over batch
        
        # Static β value (if provided)
        if kl_factor is not None:
            beta_value = kl_factor
        else:
            # Initialize moving averages
            if self.mse_avg is None or self.kl_avg is None:
                self.mse_avg = mse_loss.item()
                self.kl_avg = kl_loss.item()
            
            # Update moving averages with smoothing
            self.mse_avg = self.smooth_factor * self.mse_avg + (1 - self.smooth_factor) * mse_loss.item()
            self.kl_avg = self.smooth_factor * self.kl_avg + (1 - self.smooth_factor) * kl_loss.item()
            
            # Dynamic beta to balance the magnitudes
            beta_value = self.mse_avg / (self.kl_avg + 1e-8)  # Avoid division by zero

        loss = mse_loss + beta_value * kl_loss
        return loss, mse_loss, kl_loss, beta_value


class NLLLoss(nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, data, out):
        """Run the forward pass and compute the loss and accuracy.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input batch of data.

        Returns
        -------
        loss : float
            The loss on the given data batch.
        acc : float
            The accuracy on the current data batch.
        """

        # Loss
        loss = nnf.nll_loss(out, data.y)
        # Accuracy
        pred = out.argmax(dim=1)
        correct = float(pred.eq(data.y).sum().item())
        acc = correct / len(data.y)
        return loss, acc