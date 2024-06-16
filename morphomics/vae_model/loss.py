import torch

class Loss(torch.nn.Module):
    def __init__(self, kl_factor = 1.):
        super(Loss, self).__init__()
        self.kl_factor = kl_factor
        
    def forward(self, x, out, z_mean, z_log_var):
        # Compute the reconstruction loss (mse)
        x_expanded = x.unsqueeze(0).expand(*out.shape)
        l2 = torch.norm(out - x_expanded, dim=-1, p=2)
        mse = torch.mean(l2)
        # Compute the KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())

        return mse + self.kl_factor*kl_loss