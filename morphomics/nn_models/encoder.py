import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim = 2, hidden_dimensions=[16, 8], batch_layer_norm = False,
                 activation = nn.ReLU):
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
            if batch_layer_norm and i < 2 and 2<self.num_layers:
                 layers.append(nn.BatchNorm1d(self.layer_dimensions[i + 1])),  # Batch Normalization
            if batch_layer_norm and i == self.num_layers - 2:
                layers.append(nn.LayerNorm(self.layer_dimensions[i + 1])),  # Layer Normalization
            # Add a ReLU layer, except after the last Linear layer
            if i < self.num_layers - 1:
                layers.append(activation())
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
    

class EncoderCNN(nn.Module):
    def __init__(self, input_dim, 
                 input_channels, 
                 latent_dim=2, 
                 hidden_dims=[8, 16, 32], 
                 stride=2,
                 batch_layer_norm=False):
        super(EncoderCNN, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        layers = []

        # Convolutional layers
        in_channels = input_channels
        for h_dim in hidden_dims:
            layers.append(nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=stride, padding=1))
            if batch_layer_norm:
                layers.append(nn.BatchNorm2d(h_dim))
            layers.append(nn.ReLU())
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*layers)

        # Dynamically calculate the final feature map size after convolutions
        final_height, final_width = self._get_conv_output(input_dim)
        self.flatten_size = hidden_dims[-1] * final_height * final_width

        # Fully connected layers for mean and log variance
        self.fc_mean = nn.Linear(self.flatten_size, latent_dim)
        self.fc_log_var = nn.Linear(self.flatten_size, latent_dim)

    def _get_conv_output(self, input_dim):
        # Dummy tensor to calculate the final output size after convolutions
        x = torch.ones(1, 1, input_dim, input_dim)
        x = self.encoder(x)
        return x.shape[2], x.shape[3]  # Height and width after all convolutions

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)  # Flatten before FC layers
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        return z_mean, z_log_var

