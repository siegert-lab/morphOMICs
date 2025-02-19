import torch
import torch.nn as nn
from .encoder import Encoder, EncoderCNN
from .decoder import Decoder, DecoderCNN

class VAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim = 2, 
                 encoder_hidden_dimensions = [16, 8, 2], decoder_hidden_dimensions = [2, 8, 16], 
                 batch_layer_norm = False,
                 activation = nn.ReLU):
        super(VAE, self).__init__()
        
        # Initialize the encoder and decoder
        self.encoder = Encoder(input_dim = input_dim, 
                               latent_dim = latent_dim,
                                hidden_dimensions = encoder_hidden_dimensions,
                                batch_layer_norm=batch_layer_norm,
                                activation=activation)
        
        self.decoder = Decoder(latent_dim = latent_dim,
                               output_dim = input_dim,
                               hidden_dimensions = decoder_hidden_dimensions,
                               batch_layer_norm=batch_layer_norm,
                               activation=activation)
        
    def sample(self, z_mean, z_log_var, sample_size):
        # Generate a random sample
        device = z_mean.device
        epsilon = torch.randn(sample_size, *z_mean.shape).to(device)
        # Re-parameterization trick
        z = z_mean + torch.exp(0.5 * z_log_var) * epsilon
        
        return z
    
    def forward(self, x, sample_size):
        # Pass the input through the encoder
        z_mean, z_log_var = self.encoder(x)
        # Sample from the distribution
        z = self.sample(z_mean, z_log_var, sample_size)
        # Pass the sample through the decoder
        out = self.decoder(z)
        return out, z_mean, z_log_var      
    

class VAECNN(VAE):
    def __init__(self, input_dim, latent_dim = 2,
                 encoder_hidden_dimensions = [8, 16, 32], decoder_hidden_dimensions = [32, 16, 8], 
                 batch_layer_norm = False):
        # Instead of calling super() which initializes VAE with Encoder and Decoder, we use the customized CNN versions.
        super(VAECNN, self).__init__(input_dim, latent_dim, 
                                     encoder_hidden_dimensions, decoder_hidden_dimensions, 
                                     batch_layer_norm)
        
        # Redefine encoder to use EncoderCNN
        self.encoder = EncoderCNN(input_dim=input_dim, 
                                   input_channels=1,  # Example: for grayscale images
                                   latent_dim=latent_dim,
                                   hidden_dims=encoder_hidden_dimensions,
                                   stride=2,
                                   batch_layer_norm=batch_layer_norm)

        self.decoder = DecoderCNN(latent_dim=latent_dim,
                                   output_dim=input_dim,
                                   hidden_dims=decoder_hidden_dimensions,
                                   stride=2,
                                   batch_layer_norm=batch_layer_norm)
