import torch
from .encoder import Encoder
from .decoder import Decoder

class VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim = 2, encoder_hidden_dimensions = [16, 8, 2], decoder_hidden_dimensions = [2, 8, 16]):
        super(VAE, self).__init__()
        
        # Initialize the encoder and decoder
        self.encoder = Encoder(input_dim = input_dim, 
                               latent_dim = latent_dim,
                                hidden_dimensions = encoder_hidden_dimensions)
        
        self.decoder = Decoder(latent_dim = latent_dim,
                               output_dim = input_dim,
                               hidden_dimensions = decoder_hidden_dimensions)
        
    def sample(self, z_mean, z_log_var, sample_size):
        # Generate a random sample
        epsilon = torch.randn(sample_size, *z_mean.shape)
        # Re-parameterization trick
        z = z_mean + torch.exp(z_log_var / 2) * epsilon
        
        return z
    
    def forward(self, x, sample_size):
        # Pass the input through the encoder
        z_mean, z_log_var = self.encoder(x)
        
        # Sample from the distribution
        z = self.sample(z_mean, z_log_var, sample_size)
        
        # Pass the sample through the decoder
        out = self.decoder(z)

        return out, z_mean, z_log_var    