import torch
from torch import nn
from torch.nn import functional as nnf


class VAELoss(nn.Module):
    def __init__(self, kl_factor = 1.):
        super(VAELoss, self).__init__()
        self.kl_factor = kl_factor
        
    def forward(self, x, out, z_mean, z_log_var):
        # Compute the reconstruction loss (mse)
        x_expanded = x.unsqueeze(0).expand(*out.shape)
        l2 = torch.norm(out - x_expanded, dim = -1, p=2)
        mse = torch.mean(l2)
        # Compute the KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim = -1)
        kl_loss = torch.mean(kl_loss)
        return mse + self.kl_factor*kl_loss, mse
    

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