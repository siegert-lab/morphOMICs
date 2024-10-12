import torch
import torch.nn as nn


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dimensions = [8, 16], batch_layer_norm = False,
                activation = nn.ReLU):

        super(Decoder, self).__init__()
        
        # Define the layer dimensions
        self.layer_dimensions = [latent_dim] + hidden_dimensions + [output_dim]
        self.num_layers = len(self.layer_dimensions) - 1 
        # Initialize an empty list to hold the layers
        layers = []

        # Loop through the layer dimensions to create Linear and ReLU layers
        for i in range(self.num_layers):
            # Add a Linear layer
            layers.append(nn.Linear(self.layer_dimensions[i], self.layer_dimensions[i + 1]))
            # if batch_layer_norm and i < 2 and 2<self.num_layers:
            #      layers.append(nn.BatchNorm1d(self.layer_dimensions[i + 1])),  # Batch Normalization
            if batch_layer_norm and i == self.num_layers - 2:
                layers.append(nn.LayerNorm(self.layer_dimensions[i + 1])),  # Layer Normalization
            # Add a ReLU layer, except after the last Linear layer
            if i < len(self.layer_dimensions) - 2:
                layers.append(activation())
        # Create the Sequential model with the layers
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        out = self.model(z)
        return out
    
# batch_norm = nn.BatchNorm1d(2)
# reshaped_output = sampled_output.view(-1, 2)  # Shape [96, 2]
# normalized_output = batch_norm(reshaped_output)  # Shape [96, 2]

    

