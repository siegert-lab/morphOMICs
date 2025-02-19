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
class DecoderCNN(nn.Module):
    def __init__(self, latent_dim, 
                 output_dim, 
                 channel_dim=1,
                 hidden_dims=[32, 16, 8],
                 s_start=None,
                 stride=2, 
                 kernel_size=3,
                 padding=1,
                 output_padding=0,
                 batch_layer_norm=False):
        """
        Parameters:
          latent_dim: Dimension of the latent vector.
          output_dim: Desired final spatial size (assumed square, e.g. 100 for 100x100).
          channel_dim: Number of channels in the output (e.g., 1 for grayscale, 3 for RGB).
          hidden_dims: List of channels for the decoder layers.
          s_start: Optional starting spatial size; if None, it's computed heuristically.
          stride, kernel_size, padding, output_padding: Parameters for ConvTranspose2d layers.
          batch_layer_norm: Whether to include BatchNorm2d after each ConvTranspose2d.
        """
        super(DecoderCNN, self).__init__()
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.channel_dim = channel_dim
        self.hidden_dims = hidden_dims
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.output_padding = output_padding

        # Number of layers is the length of hidden_dims.
        num_layers = len(hidden_dims)
        # Heuristically compute s_start if not provided.
        if s_start is None:
            self.s_start = max(1, output_dim // (stride ** num_layers))
        else:
            self.s_start = s_start

        # Fully connected layer to project latent vector into a flattened feature map.
        self.fc = nn.Linear(latent_dim, hidden_dims[0] * self.s_start * self.s_start)
        # Reshape layer to convert the flattened vector into a tensor.
        self.reshape = nn.Unflatten(1, (hidden_dims[0], self.s_start, self.s_start))
        
        # Build decoder layers.
        layers = []
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            layers.append(nn.ConvTranspose2d(in_channels, h_dim,
                                             kernel_size=kernel_size,
                                             stride=stride,
                                             padding=padding,
                                             output_padding=output_padding))
            if batch_layer_norm:
                layers.append(nn.BatchNorm2d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_channels = h_dim
        
        # Final output layer to obtain the desired channel dimension.
        layers.append(nn.ConvTranspose2d(in_channels, channel_dim,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         output_padding=output_padding))
        # Use Tanh activation so that the pixel values are in the range [-1, 1].
        layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*layers)
        
        # Final upsampling layer to force the output size exactly to (output_dim, output_dim).
        self.upsample = nn.Upsample(size=(output_dim, output_dim), mode='bilinear', align_corners=False)

    def forward(self, z):
        """
        Expects input tensor z with shape [nb_samples, batch_size, latent_dim].
        Returns output of shape [nb_samples, batch_size, channel_dim, output_dim, output_dim],
        with pixel values in [-1, 1].
        """
        nb_samples, batch_size, latent_dim = z.shape
        # Flatten nb_samples and batch_size for processing.
        z = z.view(nb_samples * batch_size, latent_dim)
        x = self.fc(z)
        x = self.reshape(x)
        x = self.decoder(x)
        x = self.upsample(x)
        # Reshape back to [nb_samples, batch_size, channel_dim, output_dim, output_dim]
        x = x.view(nb_samples, batch_size, self.channel_dim, self.output_dim, self.output_dim)
        return x