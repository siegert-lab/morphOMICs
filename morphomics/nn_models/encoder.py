import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dimensions=[16, 8]):
        super(Encoder, self).__init__()
        
        # Define the layer dimensions
        # time 2 laten dim because we need the mean and the variance of the dist.
        self.layer_dimensions = [input_dim] + hidden_dimensions + [2*latent_dim]
        self.latent_dim = int(latent_dim)

        self.num_layers = len(self.layer_dimensions) - 1 
        # Initialize an empty list to hold the layers
        layers = []
        # Loop through the layer dimensions to create Linear and ReLU layers
        for i in range(self.num_layers):
            # Add a Linear layer
            layers.append(nn.Linear(self.layer_dimensions[i], self.layer_dimensions[i + 1]))
            
            # Add a ReLU layer, except after the last Linear layer
            if i < len(self.layer_dimensions) - 2:
                layers.append(nn.ReLU())
        # Create the Sequential model with the layers
        self.model = nn.Sequential(*layers)
        
        self.mean = nn.Linear(self.layer_dimensions[-1], self.latent_dim)
        self.log_var = nn.Linear(self.layer_dimensions[-1], self.latent_dim)
        
    def forward(self, x):
        # Pass the input through the model
        z_dist = self.model(x)
        z_mean = self.mean(z_dist)
        z_log_var = self.log_var(z_dist)
        return  z_mean, z_log_var