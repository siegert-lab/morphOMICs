import torch

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dimensions = [8, 16]):
        super(Decoder, self).__init__()
        
        # Define the layer dimensions
        self.layer_dimensions = [latent_dim] + hidden_dimensions + [output_dim]
        self.num_layers = len(self.layer_dimensions) - 1 
        # Initialize an empty list to hold the layers
        layers = []

        # Loop through the layer dimensions to create Linear and ReLU layers
        for i in range(len(self.layer_dimensions) - 1):
            # Add a Linear layer
            layers.append(torch.nn.Linear(self.layer_dimensions[i], self.layer_dimensions[i + 1]))
            
            # Add a ReLU layer, except after the last Linear layer
            if i < len(self.layer_dimensions) - 2:
                layers.append(torch.nn.ReLU())
        print(layers)
        # Create the Sequential model with the layers
        self.model = torch.nn.Sequential(*layers)
        

    def forward(self, z):
        # Pass the input through the model
        out = self.model(z)
        return out